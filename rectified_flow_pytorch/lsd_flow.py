from contextlib import nullcontext

import torch
from torch import tensor
from torch.nn import Module, Sequential, Linear, SiLU
import torch.nn.functional as F
from torch.func import jvp

from einops import reduce


def exists(v):
    return v is not None


def xnor(x, y):
    return not (x ^ y)


def default(v, d):
    return v if exists(v) else d


def identity(t):
    return t


def append_dims(t, dims):
    shape = t.shape
    ones = (1,) * dims
    return t.reshape(*shape, *ones)


def divisible_by(num, den):
    return (num % den) == 0


class LsdFlow(Module):
    def __init__(
        self,
        model: Module,
        *,
        data_shape = None,
        normalize_data_fn = identity,
        unnormalize_data_fn = identity,
        diag_prob = 0.75,
        eps = 1e-3,
        noise_std_dev = 1.0,
        accept_cond = False,
        use_learned_loss_weight = False,
        loss_weight_hidden_dim = 64,
    ):
        super().__init__()
        self.model = model
        self.data_shape = data_shape

        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn

        assert 0.0 <= diag_prob <= 1.0
        self.diag_prob = diag_prob
        self.eps = eps
        self.noise_std_dev = noise_std_dev
        self.accept_cond = accept_cond

        self.use_learned_loss_weight = use_learned_loss_weight
        if use_learned_loss_weight:
            self.loss_weight_mlp = Sequential(
                Linear(2, loss_weight_hidden_dim),
                SiLU(),
                Linear(loss_weight_hidden_dim, loss_weight_hidden_dim),
                SiLU(),
                Linear(loss_weight_hidden_dim, 1),
            )
        else:
            self.loss_weight_mlp = None

        self.register_buffer("dummy", tensor(0), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def get_noise(self, batch_size = 1, data_shape = None):
        device = self.device
        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), "shape of the data must be passed in, or set at init or during training"
        noise = torch.randn((batch_size, *data_shape), device = device) * self.noise_std_dev
        return noise

    def sample_times(self, batch):
        t = torch.rand((batch,), device = self.device)
        if self.eps > 0:
            t = t.clamp(min = self.eps, max = 1.0 - self.eps)
        return t

    def sample_triangle_times(self, batch):
        tmin = self.eps
        tmax = 1.0 - self.eps
        u1 = torch.rand((batch,), device = self.device) * (tmax - tmin) + tmin
        u2 = torch.rand((batch,), device = self.device) * (tmax - tmin) + tmin
        s = torch.minimum(u1, u2)
        t = torch.maximum(u1, u2)
        return s, t

    def sample_diag_mask(self, batch):
        if self.diag_prob <= 0.0:
            return torch.zeros((batch,), device = self.device, dtype = torch.bool)
        if self.diag_prob >= 1.0:
            return torch.ones((batch,), device = self.device, dtype = torch.bool)
        return torch.rand((batch,), device = self.device) < self.diag_prob

    def predict_velocity(self, x, s, t, cond = None):
        if self.accept_cond:
            return self.model(x, s, t, cond)
        return self.model(x, s, t)

    def flow_map(self, x_s, s, t, cond = None):
        delta = append_dims(t - s, x_s.ndim - 1)
        return x_s + delta * self.predict_velocity(x_s, s, t, cond)

    def weight_loss_per_sample(self, loss, t, s):
        if not self.use_learned_loss_weight or not self.loss_weight_mlp:
            return loss
        ts = torch.stack((s, t), dim = -1)
        log_var = self.loss_weight_mlp(ts).squeeze(-1)
        return torch.exp(-log_var) * loss + log_var

    def forward(self, data, *, return_loss_breakdown = False, cond = None, noise = None):
        assert xnor(self.accept_cond, exists(cond))

        data = self.normalize_data_fn(data)
        b, _ = data.shape[0], data.device
        self.data_shape = default(self.data_shape, data.shape[1:])
        ndim = data.ndim

        if not exists(noise):
            noise = torch.randn_like(data) * self.noise_std_dev

        diag_mask = self.sample_diag_mask(b)
        off_mask = ~diag_mask

        diag_count = diag_mask.sum()
        off_count = off_mask.sum()
        total_count = diag_count + off_count

        maybe_cond = cond if self.accept_cond else None

        # diagonal loss
        t_diag = self.sample_times(b)
        x0 = noise
        x1 = data
        t_diag_view = append_dims(t_diag, ndim - 1)
        it = x0 + t_diag_view * (x1 - x0)
        it_dot = x1 - x0
        pred = self.predict_velocity(it, t_diag, t_diag, maybe_cond)
        diag_per_sample = reduce(F.mse_loss(pred, it_dot, reduction = "none"), "b ... -> b", "mean")
        diag_per_sample = self.weight_loss_per_sample(diag_per_sample, t_diag, t_diag)
        diag_loss = (diag_per_sample * diag_mask).sum() / diag_count.clamp_min(1)

        # off-diagonal loss
        s_off, t_off = self.sample_triangle_times(b)
        s_off_view = append_dims(s_off, ndim - 1)
        is_ = x0 + s_off_view * (x1 - x0)

        def flow_map_fn(x, s, t):
            return self.flow_map(x, s, t, maybe_cond)

        tangents = (
            torch.zeros_like(is_),
            torch.zeros_like(s_off),
            torch.ones_like(t_off),
        )
        x_hat, dxt = jvp(flow_map_fn, (is_, s_off, t_off), tangents)

        x_hat = x_hat.detach()
        with torch.no_grad():
            teacher = self.predict_velocity(x_hat, t_off, t_off, maybe_cond)

        residual = dxt - teacher
        off_per_sample = reduce(residual.pow(2), "b ... -> b", "mean")
        off_per_sample = self.weight_loss_per_sample(off_per_sample, t_off, s_off)
        off_loss = (off_per_sample * off_mask).sum() / off_count.clamp_min(1)

        total_loss = torch.where(
            total_count > 0,
            (diag_loss * diag_count + off_loss * off_count) / total_count,
            diag_loss + off_loss,
        )

        if return_loss_breakdown:
            return total_loss, (diag_loss.detach(), off_loss.detach())
        return total_loss

    @torch.no_grad()
    def slow_sample(self, steps = 4, batch_size = 1, noise = None, data_shape = None, cond = None):
        assert steps >= 1

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), "shape of the data must be passed in, or set at init or during training"

        device = self.device
        maybe_cond = (cond,) if self.accept_cond else ()

        if not exists(noise):
            noise = self.get_noise(batch_size = batch_size, data_shape = data_shape)

        times = torch.linspace(0.0, 1.0, steps + 1, device = device)[:-1]
        delta = 1.0 / steps

        denoised = noise

        for time in times:
            time = time.expand(batch_size)
            next_time = time + delta
            next_time = next_time.expand(batch_size)
            denoised = self.flow_map(denoised, time, next_time, *maybe_cond)

        return self.unnormalize_data_fn(denoised)

    def sample(self, batch_size = None, data_shape = None, requires_grad = False, cond = None, noise = None, steps = 1):
        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), "shape of the data must be passed in, or set at init or during training"

        maybe_cond = ()
        if self.accept_cond:
            batch_size = cond.shape[0]
            maybe_cond = (cond,)

        batch_size = default(batch_size, 1)

        if steps > 1:
            return self.slow_sample(steps = steps, batch_size = batch_size, data_shape = data_shape, cond = cond, noise = noise)

        assert xnor(self.accept_cond, exists(cond))

        device = self.device
        context = nullcontext if requires_grad else torch.no_grad

        if not exists(noise):
            noise = self.get_noise(batch_size = batch_size, data_shape = data_shape)

        start = torch.zeros(batch_size, device = device)
        end = torch.ones(batch_size, device = device)
        with context():
            data = self.flow_map(noise, start, end, *maybe_cond)

        return self.unnormalize_data_fn(data)
