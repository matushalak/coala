import torch

from coala.active.foveation import overlay_fixation_marker, perceptual_foveate, render_foveation_bundle


def test_perceptual_foveation_preserves_shape_and_range():
    image = torch.rand(3, 96, 96)
    rendered = perceptual_foveate(
        image,
        fixation_x=0.4,
        fixation_y=0.6,
        kerw_coef=0.06,
        fovea_radius_ratio=0.12,
        ring_spacing=0.2,
    )

    assert rendered.shape == image.shape
    assert torch.all((0.0 <= rendered) & (rendered <= 1.0))


def test_perceptual_foveation_keeps_center_sharper_than_corner():
    image = torch.zeros(1, 64, 64)
    image[:, 30:34, 30:34] = 1.0
    image[:, :4, :4] = 1.0

    rendered = perceptual_foveate(
        image,
        fixation_x=0.5,
        fixation_y=0.5,
        kerw_coef=0.08,
        fovea_radius_ratio=0.10,
        ring_spacing=0.2,
    )

    center_energy = rendered[:, 30:34, 30:34].mean()
    corner_energy = rendered[:, :4, :4].mean()
    assert center_energy > corner_energy


def test_parafoveal_noise_perturbs_periphery_more_than_fixation_center():
    torch.manual_seed(0)
    image = torch.full((1, 64, 64), 0.5)
    noiseless = perceptual_foveate(
        image,
        fixation_x=0.5,
        fixation_y=0.5,
        kerw_coef=0.06,
        fovea_radius_ratio=0.12,
        ring_spacing=0.2,
        parafoveal_noise_std=0.0,
    )
    torch.manual_seed(0)
    noisy = perceptual_foveate(
        image,
        fixation_x=0.5,
        fixation_y=0.5,
        kerw_coef=0.06,
        fovea_radius_ratio=0.12,
        ring_spacing=0.2,
        parafoveal_noise_std=0.2,
    )

    center_delta = (noisy[:, 28:36, 28:36] - noiseless[:, 28:36, 28:36]).abs().mean()
    corner_delta = (noisy[:, :8, :8] - noiseless[:, :8, :8]).abs().mean()
    assert corner_delta > center_delta


def test_render_bundle_contains_expected_views():
    image = torch.rand(1, 28, 28)
    bundle = render_foveation_bundle(
        image,
        fixation_x=0.5,
        fixation_y=0.5,
        kerw_coef=0.06,
        fovea_radius_ratio=0.12,
        ring_spacing=0.2,
        parafoveal_noise_std=0.1,
        magnif_fov_ratio=0.22,
        magnif_k_ratio=0.22,
        cover_ratio=1.0,
    )

    assert set(bundle.keys()) == {"original", "overlay", "perceptual", "combined"}
    assert bundle["original"].shape == (1, 28, 28)
    assert bundle["overlay"].shape == (3, 28, 28)
    assert bundle["perceptual"].shape == (3, 28, 28)
    assert bundle["combined"].shape == (3, 28, 28)


def test_overlay_fixation_marker_returns_rgb():
    image = torch.rand(1, 28, 28)
    overlay = overlay_fixation_marker(image, fixation_x=0.5, fixation_y=0.5, marker_radius=3)

    assert overlay.shape == (3, 28, 28)
