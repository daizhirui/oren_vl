"""Evaluator for :class:`VlNetwork`: measure predicted vs ground-truth VL features.

Reports L1 (MAE), L2 (MSE / RMSE), and cosine-similarity metrics for the final prediction (`vl`) and the
prior branch (`vl_prior`, when the field has one). Two entry points (no mesh / slice extraction -- features
have no iso-surface to march):

    - :meth:`evaluate_test_set` -- load `(points, gt_features)` from `.npy` files in `test_set_dir`
      (mirrors :meth:`EvaluatorBase.sdf_and_grad_metrics`'s file-layout convention).
    - :meth:`evaluate_dataset` -- iterate any iterable of :class:`VlFrame` (e.g.
      :class:`oren.dataset.vl_feature_dataset.DataLoader`) and aggregate metrics across all valid points.

Metric keys, per output field (`vl`, optionally `vl_prior`):

    - `l1`                -- mean absolute element-wise error.
    - `l2_mse` / `l2_rmse`-- mean squared and root-mean-squared element-wise error.
    - `cosine_similarity` -- mean per-row cosine similarity in `[-1, 1]`.
    - `cosine_loss`       -- `1 - cosine_similarity` (matches :class:`oren.vl_criterion.CosineSimilarityLoss`).
    - `n`                 -- number of points contributing to the aggregate.

Standalone class, *not* a subclass of :class:`EvaluatorBase`: that base's surface (SDF sign handling,
gradient error, marching cubes) is geometry-oriented and does not transfer to per-point feature fields.
"""

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sized

import torch.nn.functional as F
from tqdm import tqdm

from oren import np, torch
from oren.frame import VlFrame
from oren.utils.config_abc import ConfigABC
from oren.vl_network import VlNetwork, VlNetworkConfig


def _new_acc() -> dict:
    """Empty rolling accumulator -- per-element sums for L1 / L2 and a per-row sum for cosine similarity."""
    return dict(n=0, l1_sum=0.0, l2_sum=0.0, cos_sim_sum=0.0)


def _accumulate(pred: torch.Tensor, gt: torch.Tensor, acc: dict) -> None:
    """Fold one batch into the rolling accumulator. Both tensors are `(N, C)` on the same device."""
    diff = pred - gt
    cos_sim = F.cosine_similarity(pred, gt, dim=-1)
    acc["n"] += pred.shape[0]
    acc["l1_sum"] += diff.abs().sum().item()
    acc["l2_sum"] += (diff * diff).sum().item()
    acc["cos_sim_sum"] += cos_sim.sum().item()


def _finalize(acc: dict, feat_dim: int) -> dict:
    """Convert rolling sums into mean metrics. Returns NaN-filled dict when no points were accumulated."""
    n = acc["n"]
    if n == 0:
        nan = float("nan")
        return dict(n=0, l1=nan, l2_mse=nan, l2_rmse=nan, cosine_similarity=nan, cosine_loss=nan)
    n_elements = n * feat_dim
    mse = acc["l2_sum"] / n_elements
    cos_sim = acc["cos_sim_sum"] / n
    return dict(
        n=n,
        l1=acc["l1_sum"] / n_elements,
        l2_mse=mse,
        l2_rmse=mse**0.5,
        cosine_similarity=cos_sim,
        cosine_loss=1.0 - cos_sim,
    )


class VlEvaluator:
    """Evaluator for :class:`VlNetwork` measuring VL-feature reconstruction quality."""

    def __init__(
        self,
        batch_size: int,
        model_cfg: VlNetworkConfig | None = None,
        model: VlNetwork | None = None,
        model_path: str | None = None,
        device: str = "cuda",
    ):
        """Build a VL evaluator.

        Args:
            batch_size: per-batch point count for model inference; non-positive disables batching.
            model_cfg: required when `model` is not provided so the network can be reconstructed.
            model: optional already-instantiated VlNetwork; if provided, `model_path` is ignored.
            model_path: optional checkpoint path; loaded via :meth:`create_model` when `model` is None.
            device: torch device for inference.
        """
        self.batch_size = batch_size
        self.model_cfg = model_cfg
        self.device = device

        if model is not None:
            self.model: VlNetwork = model.to(device)
        else:
            assert model_path is not None, "VlEvaluator: either `model` or `model_path` must be provided"
            self.model = self.create_model(model_path).to(device)
        self.model.eval()

    def create_model(self, model_path: str) -> VlNetwork:
        """Construct a :class:`VlNetwork` from `self.model_cfg` and load its weights."""
        assert self.model_cfg is not None, "VlEvaluator.create_model needs `model_cfg`"
        tqdm.write("Creating VlNetwork...")
        model = VlNetwork(self.model_cfg)
        model.to(self.device)
        tqdm.write(f"Loading model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    @torch.no_grad()
    def forward_model(
        self,
        points: torch.Tensor,
        prior_only: bool = False,
        device: Optional[str] = None,
    ) -> dict:
        """Batched VlNetwork forward.

        Args:
            points: `(N, 3)` query points in world coordinates.
            prior_only: skip the implicit branch; `vl` then equals `vl_prior`.
            device: where to place the returned tensors; defaults to `self.device`.

        Returns:
            `{'vl': (N, C) pred, 'vl_prior': (N, C) prior or None}`. Both entries are `None` if `points` is empty.
        """
        if points.shape[0] == 0:
            return dict(vl=None, vl_prior=None)

        bs = points.shape[0] if self.batch_size <= 0 else self.batch_size
        out_device = self.device if device is None else device

        pred_chunks: list[torch.Tensor] = []
        prior_chunks: list[torch.Tensor] = []
        prior_seen = False

        for i in tqdm(range(0, points.shape[0], bs), desc="Batches", ncols=120, position=1, leave=False):
            j = min(i + bs, points.shape[0])
            batch = points[i:j].to(self.device)
            out = self.model(batch, prior_only=prior_only)
            pred_chunks.append(out.pred.detach())
            if out.prior is not None:
                prior_chunks.append(out.prior.detach())
                prior_seen = True

        pred = pred_chunks[0] if len(pred_chunks) == 1 else torch.cat(pred_chunks, dim=0)
        prior = None
        if prior_seen:
            prior = prior_chunks[0] if len(prior_chunks) == 1 else torch.cat(prior_chunks, dim=0)

        return dict(vl=pred.to(out_device), vl_prior=prior.to(out_device) if prior is not None else None)

    @torch.no_grad()
    def evaluate_features(
        self,
        points: torch.Tensor,
        gt_features: torch.Tensor,
        prior_only: bool = False,
    ) -> dict:
        """Run the model on `points` and report L1 / L2 / cosine metrics vs `gt_features`.

        Args:
            points: `(N, 3)` query points in world coordinates.
            gt_features: `(N, C)` ground-truth per-point feature vectors aligned row-for-row with `points`.
            prior_only: forward with `prior_only=True` to score only the prior branch.

        Returns:
            `{'vl': {...metrics}, 'vl_prior': {...metrics}}`; `'vl_prior'` is omitted when the field has
            no prior branch.
        """
        assert points.shape[0] == gt_features.shape[0], (
            f"VlEvaluator.evaluate_features: points {tuple(points.shape)} and gt_features {tuple(gt_features.shape)}"
            f" must share the leading dim"
        )
        gt = gt_features.to(self.device).float()
        feat_dim = gt.shape[-1]
        out = self.forward_model(points, prior_only=prior_only, device=self.device)

        acc_pred = _new_acc()
        _accumulate(out["vl"].float(), gt, acc_pred)
        result: dict = dict(vl=_finalize(acc_pred, feat_dim))
        if out["vl_prior"] is not None:
            acc_prior = _new_acc()
            _accumulate(out["vl_prior"].float(), gt, acc_prior)
            result["vl_prior"] = _finalize(acc_prior, feat_dim)
        return result

    @torch.no_grad()
    def evaluate_test_set(self, test_set_dir: str, prior_only: bool = False) -> dict:
        """Load `points.npy` + `gt_features.npy` from `test_set_dir` and run :meth:`evaluate_features`."""
        points = torch.from_numpy(np.load(os.path.join(test_set_dir, "points.npy"))).float()
        gt = torch.from_numpy(np.load(os.path.join(test_set_dir, "gt_features.npy"))).float()
        return self.evaluate_features(points, gt, prior_only=prior_only)

    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataset: Iterable[VlFrame],
        prior_only: bool = False,
        frame_indices: Optional[list[int]] = None,
        num_samples_per_frame: int = -1,
        ratio: float = 1.0,
    ) -> dict:
        """Aggregate per-frame metrics into a single dataset-level result.

        Args:
            dataset: indexable / iterable yielding :class:`VlFrame` instances (e.g.
                :class:`oren.dataset.vl_feature_dataset.DataLoader`). If indexable, `frame_indices` selects
                a subset; otherwise the full iterator is consumed.
            prior_only: forward the network with `prior_only=True` to score only the prior branch.
            frame_indices: optional list of frame indices to evaluate; defaults to every frame.
            num_samples_per_frame: passed through to :meth:`VlFrame.sample_points_and_features`; `-1` keeps
                all valid points (subject to `ratio`).
            ratio: fraction of valid points sampled per frame when `num_samples_per_frame <= 0`; default
                `1.0` keeps every valid point.

        Returns:
            Same shape as :meth:`evaluate_features` -- `{'vl': {...}, 'vl_prior': {...}}` -- but the metrics
            are means over every point of every evaluated frame (point-weighted, not frame-weighted).
        """
        if frame_indices is None and isinstance(dataset, Sized):
            frame_indices = list(range(len(dataset)))

        if frame_indices is not None:
            # We've already required `dataset` to be indexable via `frame_indices`; the loop uses
            # `dataset[i]` directly so the dataset must support `__getitem__`.
            iterator: Iterable[VlFrame] = (dataset[i] for i in frame_indices)  # type: ignore[index]
            total = len(frame_indices)
        else:
            iterator = iter(dataset)
            total = None

        feat_dim: Optional[int] = None
        acc_pred = _new_acc()
        acc_prior = _new_acc()
        has_prior = False

        for frame in tqdm(iterator, total=total, desc="Frames", ncols=120):
            pts, gt = frame.sample_points_and_features(
                num_samples=num_samples_per_frame,
                ratio=ratio,
                to_world_frame=True,
                device=self.device,
            )
            if pts.numel() == 0:
                continue
            gt = gt.float()
            if feat_dim is None:
                feat_dim = gt.shape[-1]
            else:
                assert gt.shape[-1] == feat_dim, (
                    f"VlEvaluator.evaluate_dataset: inconsistent feature dim across frames "
                    f"(got {gt.shape[-1]}, expected {feat_dim})"
                )

            out = self.forward_model(pts, prior_only=prior_only, device=self.device)
            _accumulate(out["vl"].float(), gt, acc_pred)
            if out["vl_prior"] is not None:
                has_prior = True
                _accumulate(out["vl_prior"].float(), gt, acc_prior)

        # Fall back to feat_dim=1 only when zero points were seen; _finalize returns NaNs in that case so the
        # multiplier never matters.
        feat_dim = feat_dim if feat_dim is not None else 1
        result: dict = dict(vl=_finalize(acc_pred, feat_dim))
        if has_prior:
            result["vl_prior"] = _finalize(acc_prior, feat_dim)
        return result


@dataclass
class VlEvaluatorConfig(ConfigABC):
    """CLI configuration for :func:`main` -- evaluate a trained :class:`VlNetwork` checkpoint.

    Mirrors the spirit of :func:`oren.evaluator_oren.main` (load a trained model, pick one or more evaluation
    modes, write metrics to disk) but is wired through :class:`ConfigABC` so flags are auto-generated and a
    YAML can be loaded with `--config` / created with `--create-config`.

    Two evaluation modes can be enabled simultaneously:

        - `evaluate_test_set` -- score against precomputed `points.npy` / `gt_features.npy` files in
            `test_set_dir` (matches :meth:`VlEvaluator.evaluate_test_set`).
        - `evaluate_dataset` -- score against the trainer's data stream, instantiated from
            `trainer_config`'s `data` section via :func:`oren.utils.import_util.get_dataset`.
    """

    # ---- model loading ----
    # Path to the trainer YAML used to train the checkpoint. Provides both the `VlNetworkConfig` (for
    # model reconstruction) and the `DataConfig` (used when `evaluate_dataset=True`).
    trainer_config: Optional[str] = None
    # Path to the saved `state_dict` (`.pth`) for the trained `VlNetwork`.
    model_path: Optional[str] = None

    # ---- output ----
    # Directory for the metrics YAML files. If `None`, defaults to `<dirname(dirname(trainer_config))>/eval`
    # so a typical `logs/<exp>/backup/config.yaml` trainer config writes to `logs/<exp>/eval/`.
    output_dir: Optional[str] = None

    # ---- inference knobs ----
    batch_size: int = 40960
    device: str = "cuda"
    # Forward the network with `prior_only=True` to score only the prior branch (skips the implicit head).
    prior_only: bool = False

    # ---- mode: precomputed test-set ----
    evaluate_test_set: bool = False
    test_set_dir: Optional[str] = None

    # ---- mode: trainer dataset ----
    evaluate_dataset: bool = False
    # Per-frame point budget forwarded to `VlFrame.sample_points_and_features`; `-1` keeps all valid points.
    num_samples_per_frame: int = -1
    # Per-frame subsampling ratio used when `num_samples_per_frame <= 0`; `1.0` keeps every valid point.
    ratio: float = 1.0
    # Optional subset of frame indices to score; `None` means every frame in the dataset.
    frame_indices: Optional[list[int]] = None


def _dump_yaml(metrics: dict, out_path: str) -> None:
    """Write the metrics dict to `out_path` as YAML, preserving the field/metric ordering."""
    import yaml

    with open(out_path, "w") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)


def main() -> None:
    """CLI entry point: load a trained `VlNetwork` and run L1 / L2 / cosine metrics.

    Driven by :class:`VlEvaluatorConfig`. Run `python -m oren.evaluator_vl --help` for the full flag list;
    `--config <path>` loads a YAML; `--create-config <path>` writes a default YAML and exits.

    At least one of `--evaluate-test-set` / `--evaluate-dataset` must be set. Results are written as YAML
    under `output_dir` (`vl_metrics_test_set.yaml` and/or `vl_metrics_dataset.yaml`) and printed to stdout.
    """
    from oren.trainer_config import TrainerConfig
    from oren.utils.import_util import get_dataset

    parser = VlEvaluatorConfig.get_argparser()
    cfg, _ = parser.parse_known_args()

    assert cfg.trainer_config is not None, "VlEvaluator CLI: --trainer-config is required"
    assert cfg.model_path is not None, "VlEvaluator CLI: --model-path is required"

    trainer_cfg = TrainerConfig.from_yaml(cfg.trainer_config)
    assert isinstance(trainer_cfg.model, VlNetworkConfig), (
        f"VlEvaluator CLI: trainer_config's `model` must be a VlNetworkConfig; "
        f"got {type(trainer_cfg.model).__qualname__}"
    )

    output_dir = cfg.output_dir
    if output_dir is None:
        # Mirror `evaluator_oren.main`: `<logs/exp>/eval` relative to the trainer config.
        output_dir = os.path.join(os.path.dirname(os.path.dirname(cfg.trainer_config)), "eval")
        output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    evaluator = VlEvaluator(
        batch_size=cfg.batch_size,
        model_cfg=trainer_cfg.model,
        model_path=cfg.model_path,
        device=cfg.device,
    )

    if not (cfg.evaluate_test_set or cfg.evaluate_dataset):
        tqdm.write("No evaluation mode selected -- pass --evaluate-test-set and/or --evaluate-dataset.")
        return

    if cfg.evaluate_test_set:
        assert cfg.test_set_dir is not None, "VlEvaluator CLI: --test-set-dir required with --evaluate-test-set"
        metrics = evaluator.evaluate_test_set(cfg.test_set_dir, prior_only=cfg.prior_only)
        out_path = os.path.join(output_dir, "vl_metrics_test_set.yaml")
        _dump_yaml(metrics, out_path)
        tqdm.write(f"Test-set metrics: {metrics}")
        tqdm.write(f"Saved test-set metrics to {out_path}")

    if cfg.evaluate_dataset:
        dataset = get_dataset(trainer_cfg.data.dataset_name, trainer_cfg.data.dataset_args)
        metrics = evaluator.evaluate_dataset(
            dataset,
            prior_only=cfg.prior_only,
            frame_indices=cfg.frame_indices,
            num_samples_per_frame=cfg.num_samples_per_frame,
            ratio=cfg.ratio,
        )
        out_path = os.path.join(output_dir, "vl_metrics_dataset.yaml")
        _dump_yaml(metrics, out_path)
        tqdm.write(f"Dataset metrics: {metrics}")
        tqdm.write(f"Saved dataset metrics to {out_path}")


if __name__ == "__main__":
    main()
