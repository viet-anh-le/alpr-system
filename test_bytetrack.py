import numpy as np
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

tracker = ByteTrack(min_conf=0.1, track_thresh=0.45, match_thresh=0.8, track_buffer=30, frame_rate=30)
dets = np.array([[10, 10, 50, 50, 0.9, 0], [100, 100, 150, 150, 0.8, 1]], dtype=np.float32)
frame = np.zeros((200, 200, 3), dtype=np.uint8)

res = tracker.update(dets, frame)
print("Result shape:", res.shape if res is not None else "None")
print("Result:", res)
