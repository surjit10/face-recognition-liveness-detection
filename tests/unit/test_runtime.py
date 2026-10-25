import queue
import threading
import time
import numpy as np
import pytest

from faceguard.runtime import FramePacket, LatestFrameQueue


def test_latest_frame_queue_drops_oldest_waiting_frame() -> None:
    frames = LatestFrameQueue(maxsize=2)
    for sequence in range(3):
        frames.put_latest(
            FramePacket(np.zeros((2, 2, 3), dtype=np.uint8), float(sequence), sequence)
        )
    assert frames.dropped_frames == 1
    assert frames.get().sequence == 1
    frames.task_done()
    assert frames.get().sequence == 2
    frames.task_done()


def test_latest_frame_queue_invalid_maxsize() -> None:
    with pytest.raises(ValueError, match="maxsize must be positive"):
        LatestFrameQueue(maxsize=0)
    with pytest.raises(ValueError, match="maxsize must be positive"):
        LatestFrameQueue(maxsize=-1)


def test_latest_frame_queue_timeout_and_empty() -> None:
    frames = LatestFrameQueue(maxsize=2)
    assert frames.size == 0
    with pytest.raises(queue.Empty):
        frames.get(timeout=0.01)


def test_latest_frame_queue_stress_and_bounded_memory() -> None:
    """Stress test with high-frequency producer and slow consumer.
    
    Verifies that queue size is strictly bounded by maxsize, dropped frames
    are accurately tracked, and no memory accumulation occurs.
    """
    max_size = 3
    total_frames = 200
    frames = LatestFrameQueue(maxsize=max_size)

    produced_sequences = []
    consumed_sequences = []
    stop_event = threading.Event()

    def producer() -> None:
        for seq in range(total_frames):
            dummy = np.zeros((10, 10, 3), dtype=np.uint8)
            frames.put_latest(FramePacket(dummy, time.monotonic() * 1000.0, seq))
            produced_sequences.append(seq)
            assert frames.size <= max_size
            time.sleep(0.0005)  # 2000 fps burst
        stop_event.set()

    def consumer() -> None:
        while not stop_event.is_set() or frames.size > 0:
            try:
                packet = frames.get(timeout=0.02)
                consumed_sequences.append(packet.sequence)
                frames.task_done()
                time.sleep(0.005)  # 200 fps consumer (slower than producer)
            except queue.Empty:
                continue

    t_prod = threading.Thread(target=producer)
    t_cons = threading.Thread(target=consumer)

    t_prod.start()
    t_cons.start()
    t_prod.join(timeout=5.0)
    t_cons.join(timeout=5.0)

    assert len(produced_sequences) == total_frames
    assert len(consumed_sequences) > 0
    assert frames.dropped_frames > 0
    assert len(consumed_sequences) + frames.dropped_frames == total_frames
    assert frames.size <= max_size
