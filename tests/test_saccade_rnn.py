import torch

from coala.active.saccade_rnn import MNISTSaccadeRNN, SigReg, compute_ssl_losses


def test_saccade_rnn_forward_and_loss_are_finite():
    torch.manual_seed(0)
    model = MNISTSaccadeRNN(
        v1_features=8,
        rep_features=4,
        saccade_features=3,
        blur_sigma=0.5,
    )
    imgs = torch.randn(2, 1, 28, 28)

    outputs = model(imgs, num_steps=4, return_retina_sequence=True)
    loss, metrics = compute_ssl_losses(
        outputs,
        sigreg=SigReg(random_projections=8, max_samples=16),
        sigreg_weight=0.1,
        saccade_var_weight=0.1,
    )

    assert outputs["representations"].shape == (2, 4, 4)
    assert outputs["saccades"].shape == (2, 4, 2)
    assert outputs["retina_sequence"].shape == (2, 4, 1, 28, 28)
    assert torch.all((0.0 <= outputs["saccades"]) & (outputs["saccades"] <= 1.0))
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())

    loss.backward()
