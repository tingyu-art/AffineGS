"""INR-based velocity field estimators."""

from .frames import events_to_est, events_to_voxel_grid
from .inr import MLP_Conv, train_func
