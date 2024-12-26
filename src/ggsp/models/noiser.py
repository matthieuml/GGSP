import torch

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
