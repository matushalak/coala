from __future__ import annotations

import torch

from coala.JEPA2model import JEPA2, SIGReg, StableGlobalResponseNorm, StableSparseGlobalResponseNorm


def test_sigreg_penalizes_collapsed_features_more_than_gaussian():
    torch.manual_seed(0)
    sigreg = SIGReg(num_slices=16, knots=9, max_samples=256)
    gaussian = torch.randn(256, 32)
    collapsed = torch.zeros(256, 32)

    gaussian_loss = sigreg(gaussian)
    collapsed_loss = sigreg(collapsed)

    assert collapsed_loss > gaussian_loss


def test_jepa2_compute_losses_returns_metrics():
    torch.manual_seed(0)
    model = JEPA2(
        num_filters=32,
        lr=1e-3,
        mask_ratio=0.6,
        patch_size=4,
        use_skip=False,
        sigreg_num_slices=8,
        sigreg_knots=9,
        sigreg_max_samples=256,
    )
    imgs = torch.randn(2, 1, 28, 28)

    total_loss, distill_loss, sigreg_loss, recon_loss, metrics = model.compute_losses(
        imgs,
        return_metrics=True,
    )

    assert total_loss.ndim == 0
    assert distill_loss.ndim == 0
    assert sigreg_loss.ndim == 0
    assert recon_loss.ndim == 0
    assert "feat4_loss" in metrics
    assert "feat28_visible_loss" in metrics
    assert "feat14_sigreg" in metrics


def test_jepa2_replaces_grn_with_stable_variants():
    model = JEPA2(
        num_filters=32,
        lr=1e-3,
        mask_ratio=0.6,
        patch_size=4,
        use_skip=False,
    )

    assert any(isinstance(module, StableGlobalResponseNorm) for module in model.modules())
    assert any(isinstance(module, StableSparseGlobalResponseNorm) for module in model.modules())


def test_jepa2_feature_visualizations_cover_all_scales():
    torch.manual_seed(0)
    model = JEPA2(
        num_filters=32,
        lr=1e-3,
        mask_ratio=0.6,
        patch_size=4,
        use_skip=False,
        sigreg_num_slices=8,
        sigreg_knots=9,
        sigreg_max_samples=256,
    )
    imgs = torch.randn(2, 1, 28, 28)

    visualizations = model.feature_visualizations(imgs)

    assert set(visualizations) == {"context", "feat28", "feat14", "feat7", "feat4"}
    assert visualizations["context"].shape == (4, 3, 28, 28)
    for key in ("feat28", "feat14", "feat7", "feat4"):
        assert visualizations[key].shape == (6, 3, 28, 28)
