"""
Compatibility entrypoint for the Gaussian flow implementation.

The implementation is split across ``gaussian_flow_core`` so existing code can
keep importing from ``gaussian_flow`` without changes.
"""

from gaussian_flow_core import (
    GaussianFlowConfig,
    GaussianFlowResult,
    GaussianSpatialIndexer,
    GaussianVelocityField_Tiled,
    adaptive_density_control,
    aggregate_block_map_over_gaussian_support,
    append_gaussians,
    block_indices_to_centers,
    block_reduce_map,
    build_event_density_map,
    compute_flow_curvature,
    create_optimizer,
    decide_phase,
    flow_regularizer,
    get_rebuild_interval,
    precompute_clone_candidates,
    prune_gaussians,
    prune_tensor_in_optimizer,
    replace_tensor_to_optimizer,
    render_gaussian_flow,
    robust_normalize,
    run_gaussian_flow,
    sample_block_map_at_gaussians,
    sample_maps_at_gaussians,
    train_gaussian_flow,
)

__all__ = [
    "GaussianFlowConfig",
    "GaussianFlowResult",
    "GaussianSpatialIndexer",
    "GaussianVelocityField_Tiled",
    "adaptive_density_control",
    "aggregate_block_map_over_gaussian_support",
    "append_gaussians",
    "block_indices_to_centers",
    "block_reduce_map",
    "build_event_density_map",
    "compute_flow_curvature",
    "create_optimizer",
    "decide_phase",
    "flow_regularizer",
    "get_rebuild_interval",
    "precompute_clone_candidates",
    "prune_gaussians",
    "prune_tensor_in_optimizer",
    "replace_tensor_to_optimizer",
    "render_gaussian_flow",
    "robust_normalize",
    "run_gaussian_flow",
    "sample_block_map_at_gaussians",
    "sample_maps_at_gaussians",
    "train_gaussian_flow",
]
