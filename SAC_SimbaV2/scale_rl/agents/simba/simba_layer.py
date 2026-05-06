from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class PreLNResidualBlock(nn.Module):
    hidden_dim: int
    expansion: int = 4

    def setup(self):
        self.pre_ln = nn.LayerNorm()
        self.w1 = nn.Dense(
            self.hidden_dim * self.expansion, kernel_init=nn.initializers.he_normal()
        )
        self.w2 = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal())

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        res = x
        x = self.pre_ln(x)
        x = self.w1(x)
        x = nn.relu(x)
        x = self.w2(x)
        return res + x


class LinearCritic(nn.Module):
    def setup(self):
        self.w = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        value = self.w(x).squeeze(-1)
        info = {}
        return value, info


class NormalTanhPolicy(nn.Module):
    action_dim: int
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    def setup(self):
        self.mean_w = nn.Dense(
            self.action_dim, kernel_init=nn.initializers.orthogonal(1.0)
        )
        self.std_w = nn.Dense(
            self.action_dim, kernel_init=nn.initializers.orthogonal(1.0)
        )

    def __call__(
        self,
        x: jnp.ndarray,
        temperature: float = 1.0,
    ) -> tfd.Distribution:
        mean = self.mean_w(x)
        log_std = self.std_w(x)

        # normalize log-stds for stability
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1 + nn.tanh(log_std)
        )

        # N(mu, exp(log_sigma))
        dist = tfd.MultivariateNormalDiag(
            loc=mean,
            scale_diag=jnp.exp(log_std) * temperature,
        )

        # tanh(N(mu, sigma))
        dist = tfd.TransformedDistribution(distribution=dist, bijector=tfb.Tanh())

        info = {}
        return dist, info
