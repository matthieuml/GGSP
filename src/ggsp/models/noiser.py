import torch
import torch.nn.functional as F

from ggsp.utils import extract

# forward diffusion (using the nice property)
def q_sample(
    x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None
):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

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