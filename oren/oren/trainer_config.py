from dataclasses import dataclass, field
from typing import Optional, Literal

from oren.sdf_criterion import SdfCriterionConfig, SdfCriterion
from oren.occ_criterion import OccCriterionConfig
from oren.vl_criterion import VlCriterionConfig
from oren.dataset.data_config import DataConfig
from oren.key_frame_set import KeyFrameSetConfig
from oren.sdf_network import SdfNetworkConfig, SdfNetwork
from oren.occ_network import OccNetworkConfig
from oren.vl_network import VlNetworkConfig
from oren.utils.config_abc import ConfigABC
from oren.utils.registry import get_identifier
from oren.utils.sampling import SampleRaysConfig


@dataclass
class TrainerConfig(ConfigABC):
    trainer_identifier: str = "oren.sdf_trainer.SdfTrainer"

    # ---- bookkeeping ----
    seed: int = 12345
    log_dir: str = "logs"
    exp_name: str = "oren"
    device: str = "cuda"

    # ---- data ----
    data: DataConfig = field(default_factory=DataConfig)
    bound_min: Optional[list[float]] = None
    bound_max: Optional[list[float]] = None
    key_frame_set: KeyFrameSetConfig = field(default_factory=KeyFrameSetConfig)

    # ---- model and criterion ----
    model_identifier: str = get_identifier(SdfNetwork)
    model: SdfNetworkConfig | OccNetworkConfig | VlNetworkConfig = field(default_factory=SdfNetworkConfig)
    criterion_identifier: str = get_identifier(SdfCriterion)
    criterion: SdfCriterionConfig | OccCriterionConfig | VlCriterionConfig = field(default_factory=SdfCriterionConfig)

    # ---- mode ----
    mode: Literal["scatter", "optimize"] = "optimize"
    # If True, run the optimization loop with seen frames. Otherwise, train for `offline_epochs`.
    online: bool = True
    offline_epochs: int = 10
    # Number of frames in each offline batch; only relevant when `online=False`.
    offline_batch_frames: int = 10
    offline_shuffle: bool = True

    # ---- online training ----
    num_init_frames: int = 3
    init_frame_iterations: int = 10
    num_iterations_per_frame: int = 1
    num_rays_total: int = 20480
    sample_rays: SampleRaysConfig = field(default_factory=SampleRaysConfig)

    # ---- extra sampling ----
    extra_surface_sample: bool = True
    frame_downsample: int = 100

    # ---- optimization ----
    # Number of samples (points) in each batch
    batch_size: int = 204800
    lr: float = 0.01

    # ---- gradient computation ----
    grad_method: str = "finite_difference"  # autodiff | finite_difference
    finite_difference_eps: float = 0.03

    # ---- Final evaluation and saving ----
    # number of iterations after all frames are processed, 0 means no extra iterations
    final_iterations: int = 0
    # whether to call evaluate() in the cleanup finally
    final_evaluate: bool = True
    # whether to write final.pth in the cleanup finally
    final_save_model: bool = True
    # whether to save the final mesh
    final_save_mesh: bool = True
    mesh_resolution: float = 0.0125
    mesh_iso_value: float = 0.0
    clean_mesh: bool = True
    # whether to save the final slice
    final_save_slice: bool = True
    # if None, use the center of the scene bounding box
    slice_center: Optional[list] = None
    # Dump profiling results to `<log_dir>/misc/profiling_stats.yaml` at the end of `train()`.
    # The per-timer/per-memory-profiler breakdown is only populated when `profiling=True`.
    # The headline `total_wall_time_s` is recorded unconditionally.
    final_save_profiling_stats: bool = True
    # interval to save checkpoints, -1 means no intermediate checkpoints
    ckpt_interval: int = -1

    # ---- profiling and debugging ----
    profiling: bool = False
    profiling_verbose: bool = False

    # ---- resume ----
    frozen_model_path: Optional[str] = None
