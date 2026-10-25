import os
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_dockerfile_and_deployment_files_exist():
    assert (PROJECT_ROOT / "Dockerfile").is_file()
    assert (PROJECT_ROOT / "deployment/docker-compose.yml").is_file()
    assert (PROJECT_ROOT / "deployment/run_container.sh").is_file()
    assert os.access(PROJECT_ROOT / "deployment/run_container.sh", os.X_OK)


def test_docker_compose_syntax_and_security_mounts():
    compose_path = PROJECT_ROOT / "deployment/docker-compose.yml"
    data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    assert "services" in data
    assert "faceguard" in data["services"]
    svc = data["services"]["faceguard"]

    assert svc.get("user") == "1000:1000"

    volumes = svc.get("volumes", [])
    model_mount = [v for v in volumes if "models/private" in v]
    assert len(model_mount) == 1
    # Verify read-only mount
    assert model_mount[0].endswith(":ro")
