#!/usr/bin/env python3
"""Distributed MobileNetV3-Small fine-tuning for RGB presentation-attack detection."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image, ImageOps
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.config import sha256_file  # noqa: E402
from faceguard.pad_metrics import confusion, rates, roc_auc, select_eer_threshold  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seed_everything(seed: int, rank: int) -> None:
    effective = seed + rank
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    torch.cuda.manual_seed_all(effective)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "relative_path",
        "sha256",
        "subject_id",
        "target_live",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Invalid or empty PAD manifest: {path}")
    return rows


def expanded_crop(
    image: Image.Image,
    bbox_224: tuple[float, float, float, float],
    scale: float,
    max_clipped_fraction: float,
) -> Image.Image:
    width, height = image.size
    x, y, box_width, box_height = bbox_224
    x *= width / 224.0
    box_width *= width / 224.0
    y *= height / 224.0
    box_height *= height / 224.0
    side = scale * max(box_width, box_height)
    center_x = x + box_width / 2.0
    center_y = y + box_height / 2.0
    left = math.floor(center_x - side / 2.0)
    top = math.floor(center_y - side / 2.0)
    right = math.ceil(center_x + side / 2.0)
    bottom = math.ceil(center_y + side / 2.0)
    requested_area = max(1, right - left) * max(1, bottom - top)
    clipped = (max(0, left), max(0, top), min(width, right), min(height, bottom))
    clipped_width = clipped[2] - clipped[0]
    clipped_height = clipped[3] - clipped[1]
    if clipped_width <= 0 or clipped_height <= 0:
        raise RuntimeError("PAD crop is empty")
    clipped_fraction = 1.0 - (clipped_width * clipped_height / requested_area)
    if clipped_fraction > max_clipped_fraction:
        raise RuntimeError(
            f"PAD crop clipping {clipped_fraction:.3f} exceeds "
            f"limit {max_clipped_fraction:.3f}"
        )
    return image.crop(clipped)


class PadDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        root: Path,
        rows: list[dict[str, str]],
        transform: transforms.Compose,
        crop_scale: float,
        max_clipped_fraction: float,
    ) -> None:
        self.root = root
        self.rows = rows
        self.transform = transform
        self.crop_scale = crop_scale
        self.max_clipped_fraction = max_clipped_fraction

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        path = self.root / row["relative_path"]
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                crop = expanded_crop(
                    image,
                    tuple(float(row[name]) for name in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")),
                    self.crop_scale,
                    self.max_clipped_fraction,
                )
                tensor = self.transform(crop)
        except Exception as exc:
            raise RuntimeError(f"Unable to load PAD sample {row['relative_path']}: {exc}") from exc
        label = torch.tensor(float(row["target_live"]), dtype=torch.float32)
        return tensor, label


def make_transforms(train: bool) -> transforms.Compose:
    operations: list[Any] = [transforms.Resize((224, 224), antialias=True)]
    if train:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08)],
                    p=0.35,
                ),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.08),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation without DistributedSampler's duplicate padding."""

    def __init__(self, dataset: Dataset[Any], rank: int, world_size: int) -> None:
        self.length = len(dataset)
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        return len(range(self.rank, self.length, self.world_size))


def gather_arrays(labels: list[float], scores: list[float], world_size: int) -> tuple[np.ndarray, np.ndarray]:
    payload = (labels, scores)
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, payload)
    all_labels = np.asarray([item for part, _ in gathered for item in part], dtype=np.int64)
    all_scores = np.asarray([item for _, part in gathered for item in part], dtype=np.float64)
    return all_labels, all_scores


def set_training_stage(model: nn.Module, stage: str) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if stage == "head":
        for parameter in model.classifier[-1].parameters():
            parameter.requires_grad = True
    elif stage == "finetune":
        for block in model.features[-3:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    else:
        raise RuntimeError(f"Unknown training stage: {stage}")
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def build_optimizer(model: nn.Module, stage: str) -> torch.optim.Optimizer:
    if stage == "head":
        groups = [{"params": model.classifier[-1].parameters(), "lr": 1e-3}]
    elif stage == "finetune":
        groups = [
            {"params": model.features[-3:].parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ]
    else:
        raise RuntimeError(f"Unknown optimizer stage: {stage}")
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def freeze_batchnorm_for_stage(model: nn.Module, stage: str) -> None:
    frozen_features = model.features if stage == "head" else model.features[:-3]
    for module in frozen_features.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def run_tiny_overfit(
    base_model: nn.Module,
    dataset_root: Path,
    rows: list[dict[str, str]],
    device: torch.device,
    crop_scale: float,
    max_clipped_fraction: float,
) -> dict[str, float | int]:
    """S0: prove a new final layer can learn a fixed balanced mini-cohort."""

    selected: list[dict[str, str]] = []
    for target in ("0", "1"):
        selected.extend([row for row in rows if row["target_live"] == target][:32])
    if len(selected) < 16 or {row["target_live"] for row in selected} != {"0", "1"}:
        raise RuntimeError("S0 requires at least 16 examples spanning both PAD classes")
    dataset = PadDataset(
        dataset_root,
        selected,
        make_transforms(False),
        crop_scale,
        max_clipped_fraction,
    )
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    images, targets = next(iter(loader))
    images = images.to(device)
    targets = targets.to(device)
    model = copy.deepcopy(base_model).to(device)
    set_training_stage(model, "head")
    model.eval()
    optimizer = torch.optim.AdamW(model.classifier[-1].parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images).squeeze(1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    report = {
        "samples": len(dataset),
        "steps": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
    }
    del model, images, targets
    torch.cuda.empty_cache()
    if not report["final_loss"] < report["initial_loss"] * 0.85:
        raise RuntimeError(f"S0 tiny-overfit gate failed: {report}")
    return report


@torch.no_grad()
def evaluate(
    wrapped_model: DistributedDataParallel,
    loader: DataLoader,
    device: torch.device,
    world_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    wrapped_model.eval()
    labels: list[float] = []
    scores: list[float] = []
    loss_sum = 0.0
    count = 0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = wrapped_model(images).squeeze(1)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite PAD logits during evaluation")
        loss_sum += float(criterion(logits.float(), targets).item())
        count += len(targets)
        labels.extend(targets.cpu().tolist())
        scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
    totals = torch.tensor([loss_sum, count], dtype=torch.float64, device=device)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    all_labels, all_scores = gather_arrays(labels, scores, world_size)
    return all_labels, all_scores, float(totals[0].item() / totals[1].item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--batch-size-per-gpu", type=int, default=64)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-scale", type=float, default=1.3)
    parser.add_argument("--max-clipped-fraction", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    args = parser.parse_args()

    if not 1 <= args.head_epochs <= 5:
        raise RuntimeError("--head-epochs must be in [1, 5]")
    if not 1 <= args.finetune_epochs <= 15:
        raise RuntimeError("--finetune-epochs must be in [1, 15]")
    if args.batch_size_per_gpu < 1 or args.workers_per_gpu < 0:
        raise RuntimeError("Batch size must be positive and worker count cannot be negative")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PAD fine-tuning")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (1, 2):
        raise RuntimeError(f"This Kaggle run requires 1 or 2 GPU processes; got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # CRITICAL FOR KAGGLE MULTI-GPU (Tesla T4 x2):
    # Virtualized cloud environments lack PCIe P2P direct memory access.
    # Without these environment flags, NCCL deadlocks during init_process_group.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    if rank == 0:
        print(f"[DDP] Initializing NCCL process group (world_size={world_size}, local_rank={local_rank})...", flush=True)
    dist.init_process_group(backend="nccl")
    if rank == 0:
        print("[DDP] NCCL process group initialized successfully!", flush=True)
    seed_everything(args.seed, rank)

    output = args.output_dir.resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    manifest_paths = {
        name: args.manifest_dir / f"pad_{name}.csv" for name in ("train", "val", "test")
    }
    rows = {name: read_manifest(path) for name, path in manifest_paths.items()}
    manifest_hashes = {name: sha256_file(path) for name, path in manifest_paths.items()}
    dataset_root = args.dataset_root.resolve(strict=True)

    train_dataset = PadDataset(
        dataset_root,
        rows["train"],
        make_transforms(True),
        args.crop_scale,
        args.max_clipped_fraction,
    )
    eval_datasets = {
        name: PadDataset(
            dataset_root,
            rows[name],
            make_transforms(False),
            args.crop_scale,
            args.max_clipped_fraction,
        )
        for name in ("val", "test")
    }
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    )
    eval_samplers = {
        name: DistributedEvalSampler(dataset, rank=rank, world_size=world_size)
        for name, dataset in eval_datasets.items()
    }
    common_loader = {
        "batch_size": args.batch_size_per_gpu,
        "num_workers": args.workers_per_gpu,
        "pin_memory": True,
        "persistent_workers": args.workers_per_gpu > 0,
    }
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, drop_last=True, **common_loader
    )
    eval_loaders = {
        name: DataLoader(dataset, sampler=eval_samplers[name], shuffle=False, **common_loader)
        for name, dataset in eval_datasets.items()
    }

    if rank == 0:
        print("Loading MobileNetV3-Small backbone pretrained weights...", flush=True)
    base_model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    in_features = base_model.classifier[-1].in_features
    base_model.classifier[-1] = nn.Linear(in_features, 1)

    if rank == 0:
        print("Running Stage 0 (S0) tiny-overfit sanity gate (50 steps on 32 samples)...", flush=True)
    s0_report = run_tiny_overfit(
        base_model,
        dataset_root,
        rows["train"],
        device,
        args.crop_scale,
        args.max_clipped_fraction,
    )
    if rank == 0:
        print(
            f"[✓] S0 tiny-overfit gate PASSED: loss {s0_report['initial_loss']:.4f} -> {s0_report['final_loss']:.4f}",
            flush=True,
        )
    trainable_names = set_training_stage(base_model, "head")
    base_model.to(device)
    wrapped = DistributedDataParallel(base_model, device_ids=[local_rank])
    attack_count = sum(int(row["target_live"]) == 0 for row in rows["train"])
    live_count = len(rows["train"]) - attack_count
    pos_weight = torch.tensor([attack_count / live_count], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = build_optimizer(base_model, "head")
    total_epochs = args.head_epochs + args.finetune_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.head_epochs
    )
    try:
        scaler = torch.amp.GradScaler("cuda")
    except AttributeError:  # PyTorch < 2.3
        scaler = torch.cuda.amp.GradScaler()
    config = {
        "architecture": "torchvision_mobilenet_v3_small",
        "weights": "IMAGENET1K_V1",
        "classifier_in_features": in_features,
        "label_mapping": {"1": "bona_fide", "0": "attack"},
        "world_size": world_size,
        "global_batch_size": args.batch_size_per_gpu * world_size,
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "workers_per_gpu": args.workers_per_gpu,
        "head_epochs": args.head_epochs,
        "finetune_epochs": args.finetune_epochs,
        "freeze_policy": {
            "S0": "fixed balanced mini-cohort; final linear layer only",
            "E1": "all features and classifier[:-1] frozen; final linear layer trainable",
            "E2": "features[-3:] and full classifier trainable; features[:-3] frozen",
        },
        "crop_scale": args.crop_scale,
        "max_clipped_fraction": args.max_clipped_fraction,
        "resize": "bilinear_224x224",
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "optimizer": "AdamW",
        "head_lr": 1e-3,
        "backbone_lr": 1e-4,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "early_stopping_patience": args.early_stopping_patience,
        "seed": args.seed,
        "manifest_sha256": manifest_hashes,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "s0_tiny_overfit": s0_report,
    }
    if rank == 0:
        atomic_json(output / "pad_config.json", config)

    best_acer = math.inf
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(total_epochs):
        stage = "head" if epoch < args.head_epochs else "finetune"
        if epoch == args.head_epochs:
            dist.barrier()
            selected_e1 = torch.load(
                output / "mobilenetv3_pad_best.pth",
                map_location=device,
                weights_only=False,
            )
            base_model.load_state_dict(selected_e1["model_state_dict"])
            del wrapped
            trainable_names = set_training_stage(base_model, "finetune")
            wrapped = DistributedDataParallel(base_model, device_ids=[local_rank])
            optimizer = build_optimizer(base_model, "finetune")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.finetune_epochs
            )
            try:
                scaler = torch.amp.GradScaler("cuda")
            except AttributeError:
                scaler = torch.cuda.amp.GradScaler()
            stale_epochs = 0
        train_sampler.set_epoch(epoch)
        wrapped.train()
        freeze_batchnorm_for_stage(base_model, stage)
        train_loss_sum = 0.0
        train_count = 0
        epoch_start_time = time.time()
        num_train_batches = len(train_loader)
        if rank == 0:
            print(f"\n>>> Epoch {epoch + 1}/{total_epochs} [{stage.upper()} STAGE] (Batches: {num_train_batches})", flush=True)

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = wrapped(images).squeeze(1)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite PAD training loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item()) * len(targets)
            train_count += len(targets)

            # Live batch progress on rank 0
            if rank == 0 and (batch_idx == 0 or (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == num_train_batches):
                avg_b_loss = train_loss_sum / max(1, train_count)
                pct = 100.0 * (batch_idx + 1) / num_train_batches
                elapsed = time.time() - epoch_start_time
                print(
                    f"  [Epoch {epoch+1}/{total_epochs}] Step {batch_idx+1:4d}/{num_train_batches} ({pct:5.1f}%) | "
                    f"Batch Loss: {loss.item():.4f} | Running Avg Loss: {avg_b_loss:.4f} | {elapsed:.0f}s",
                    flush=True,
                )

        totals = torch.tensor([train_loss_sum, train_count], dtype=torch.float64, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        if rank == 0:
            print(f"  [Epoch {epoch+1}/{total_epochs}] Evaluating validation split ({len(eval_loaders['val'])} batches)...", flush=True)
        val_labels, val_scores, val_loss = evaluate(
            wrapped, eval_loaders["val"], device, world_size
        )
        selection = select_eer_threshold(val_labels, val_scores)
        epoch_result = {
            "epoch": epoch + 1,
            "stage": stage,
            "train_loss": float(totals[0].item() / totals[1].item()),
            "val_loss": val_loss,
            "val_auc": roc_auc(val_labels, val_scores),
            "val_threshold": selection,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(epoch_result)
        improved = selection["acer"] < best_acer
        if improved:
            best_acer = selection["acer"]
            stale_epochs = 0
        elif stage == "finetune":
            stale_epochs += 1
        if rank == 0:
            print(
                f"  [Epoch {epoch+1}/{total_epochs}] Val Results -> Loss: {val_loss:.4f} | "
                f"AUC: {epoch_result['val_auc']:.4f} | "
                f"EER (ACER): {selection['acer']*100:.2f}% | "
                f"APCER: {selection['apcer']*100:.2f}% | BPCER: {selection['bpcer']*100:.2f}%",
                flush=True,
            )
            checkpoint = {
                "epoch": epoch + 1,
                "stage": stage,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config,
                "trainable_parameter_names": trainable_names,
                "validation": epoch_result,
            }
            atomic_checkpoint(output / "mobilenetv3_pad_last.pth", checkpoint)
            if improved:
                atomic_checkpoint(output / "mobilenetv3_pad_best.pth", checkpoint)
            atomic_json(output / "training_history.json", {"epochs": history})
            print(json.dumps(epoch_result, sort_keys=True), flush=True)
        scheduler.step()
        stop_tensor = torch.tensor(
            [int(stage == "finetune" and stale_epochs >= args.early_stopping_patience)],
            device=device,
        )
        dist.broadcast(stop_tensor, src=0)
        if bool(stop_tensor.item()):
            break

    dist.barrier()
    best = torch.load(output / "mobilenetv3_pad_best.pth", map_location=device, weights_only=False)
    base_model.load_state_dict(best["model_state_dict"])
    val_labels, val_scores, _ = evaluate(wrapped, eval_loaders["val"], device, world_size)
    threshold_report = select_eer_threshold(val_labels, val_scores)
    test_labels, test_scores, test_loss = evaluate(
        wrapped, eval_loaders["test"], device, world_size
    )
    test_counts = confusion(test_labels, test_scores, threshold_report["threshold"])
    test_report = {
        "loss": test_loss,
        "auc": roc_auc(test_labels, test_scores),
        "threshold_from_validation": threshold_report["threshold"],
        "counts": test_counts,
        **rates(test_counts),
    }

    if rank == 0:
        base_model.cpu().eval()
        onnx_path = output / "mobilenetv3_pad.onnx"
        example = torch.randn(2, 3, 224, 224)
        torch.onnx.export(
            base_model,
            example,
            onnx_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_output = session.run(None, {"input": example.numpy()})[0]
        with torch.no_grad():
            torch_output = base_model(example).numpy()
        max_abs_difference = float(np.max(np.abs(onnx_output - torch_output)))
        if max_abs_difference > 1e-4:
            raise RuntimeError(f"ONNX parity failed: max_abs_difference={max_abs_difference}")
        metrics = {
            "schema_version": 1,
            "validation_threshold": threshold_report,
            "test": test_report,
            "onnx_parity_max_abs_difference": max_abs_difference,
            "onnx_sha256": sha256_file(onnx_path),
            "best_checkpoint_sha256": sha256_file(output / "mobilenetv3_pad_best.pth"),
            "config_sha256": sha256_file(output / "pad_config.json"),
            "limitations": [
                "RGB PAD does not guarantee resistance to masks, deepfakes, or video injection.",
                "The numerical threshold is valid only for this model, crop policy, and protocol.",
            ],
        }
        atomic_json(output / "pad_metrics.json", metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
