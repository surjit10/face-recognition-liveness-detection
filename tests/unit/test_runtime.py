import numpy as np

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
