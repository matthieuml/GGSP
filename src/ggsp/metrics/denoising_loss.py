import torch
import torch.nn.functional as F
from ggsp.models import q_sample


def p_losses(
    denoise_model,
    x_start,
    t,
    cond,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    noise=None,
    loss_type="l1",
):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(
        x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise
    )
    predicted_noise = denoise_model(x_noisy, t, cond)

    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss
