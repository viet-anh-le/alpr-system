import sys
import os
import torch
sys.path.append(os.path.abspath('references/Character-Time-series-Matching/yolov5'))

original_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = _patched_load

from models.experimental import attempt_load
model = attempt_load('references/Character-Time-series-Matching/Vietnamese/object.pt', map_location='cpu')
print("Object names:", model.names)
