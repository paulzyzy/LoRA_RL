import functools
from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.jax_utils.network import Network, PRNGKey
from scale_rl.agents.simba.simba_network import (
    SimbaActor,
    SimbaCritic,
    SimbaDoubleCritic,
    SimbaTemperature,
)
from scale_rl.agents.simba.simba_update import (
    update_actor,
    update_critic,
    update_target_network,
    update_temperature,
)
from scale_rl.buffers.base_buffer import Batch

"""
The @dataclass decorator must have `frozen=True` to ensure the instance is immutable,
allowing it to be treated as a static variable in JAX.
"""


@dataclass(frozen=True)
class SimbaConfig:
    seed: int
    normalize_observation: bool
    normalize_reward: bool
    normalized_g_max: float

    load_only_param: bool
    load_param_key: bool
    load_observation_normalizer: bool
    load_reward_normalizer: bool

    learning_rate_init: float
    learning_rate_end: float
    learning_rate_decay_rate: float
    learning_rate_decay_step: int
    weight_decay: float

    actor_num_blocks: int
    actor_hidden_dim: int
    actor_bc_alpha: float

    critic_num_blocks: int
    critic_hidden_dim: int
    critic_use_cdq: bool

    target_tau: float

    temp_initial_value: float
    temp_target_entropy: float
    temp_target_entropy_coef: float

    gamma: float
    n_step: int


@functools.partial(
    jax.jit,
    static_argnames=(
        "observation_dim",
        "action_dim",
        "cfg",
    ),
)
def _init_simba_networks(
    observation_dim: int,
    action_dim: int,
    cfg: SimbaConfig,
) -> Tuple[PRNGKey, Network, Network, Network, Network]:
    fake_observations = jnp.zeros((1, observation_dim))
    fake_actions = jnp.zeros((1, action_dim))

    rng = jax.random.PRNGKey(cfg.seed)
    rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

    # When initializing the network in the flax.nn.Module class, rng_key should be passed as rngs.
    actor = Network.create(
        network_def=SimbaActor(
            num_blocks=cfg.actor_num_blocks,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=action_dim,
        ),
        network_inputs={"rngs": actor_key, "observations": fake_observations},
        tx=optax.adamw(
            learning_rate=optax.linear_schedule(
                init_value=cfg.learning_rate_init,
                end_value=cfg.learning_rate_end,
                transition_steps=cfg.learning_rate_decay_step,
            ),
            weight_decay=cfg.weight_decay,
        ),
    )

    if cfg.critic_use_cdq:
        critic_network_def = SimbaDoubleCritic(
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
        )
    else:
        critic_network_def = SimbaCritic(
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
        )

    critic = Network.create(
        network_def=critic_network_def,
        network_inputs={
            "rngs": critic_key,
            "observations": fake_observations,
            "actions": fake_actions,
        },
        tx=optax.adamw(
            learning_rate=optax.linear_schedule(
                init_value=cfg.learning_rate_init,
                end_value=cfg.learning_rate_end,
                transition_steps=cfg.learning_rate_decay_step,
            ),
            weight_decay=cfg.weight_decay,
        ),
    )

    # we set target critic's parameters identical to critic by using same rng.
    target_network_def = critic_network_def
    target_critic = Network.create(
        network_def=target_network_def,
        network_inputs={
            "rngs": critic_key,
            "observations": fake_observations,
            "actions": fake_actions,
        },
        tx=None,
    )

    temperature = Network.create(
        network_def=SimbaTemperature(cfg.temp_initial_value),
        network_inputs={
            "rngs": temp_key,
        },
        tx=optax.adamw(
            learning_rate=optax.linear_schedule(
                init_value=cfg.learning_rate_init,
                end_value=cfg.learning_rate_end,
                transition_steps=cfg.learning_rate_decay_step,
            ),
            weight_decay=cfg.weight_decay,
        ),
    )

    return rng, actor, critic, target_critic, temperature


@jax.jit
def _sample_simba_actions(
    rng: PRNGKey,
    actor: Network,
    observations: jnp.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    rng, key = jax.random.split(rng)
    dist, _ = actor(observations=observations, temperature=temperature)
    actions = dist.sample(seed=key)

    return rng, actions


@functools.partial(
    jax.jit,
    static_argnames=(
        "gamma",
        "n_step",
        "critic_use_cdq",
        "target_tau",
        "temp_target_entropy",
        "actor_bc_alpha",
    ),
)
def _update_simba_networks(
    rng: PRNGKey,
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    batch: Batch,
    gamma: float,
    n_step: int,
    actor_bc_alpha: float,
    critic_use_cdq: bool,
    target_tau: float,
    temp_target_entropy: float,
) -> Tuple[PRNGKey, Network, Network, Network, Network, Dict[str, float]]:
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    new_actor, actor_info = update_actor(
        key=actor_key,
        actor=actor,
        critic=critic,
        temperature=temperature,
        batch=batch,
        use_cdq=critic_use_cdq,
        bc_alpha=actor_bc_alpha,
    )

    new_temperature, temperature_info = update_temperature(
        temperature=temperature,
        entropy=actor_info["actor/entropy"],
        target_entropy=temp_target_entropy,
    )

    new_critic, critic_info = update_critic(
        key=critic_key,
        actor=new_actor,
        critic=critic,
        target_critic=target_critic,
        temperature=new_temperature,
        batch=batch,
        gamma=gamma,
        n_step=n_step,
        use_cdq=critic_use_cdq,
    )

    new_target_critic, target_critic_info = update_target_network(
        network=new_critic,
        target_network=target_critic,
        target_tau=target_tau,
    )

    info = {
        **actor_info,
        **critic_info,
        **target_critic_info,
        **temperature_info,
    }

    return (rng, new_actor, new_critic, new_target_critic, new_temperature, info)


class SimbaAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        cfg: SimbaConfig,
    ):
        """
        An agent that randomly selects actions without training.
        Useful for collecting baseline results and for debugging purposes.
        """

        self._observation_dim = observation_space.shape[-1]
        self._action_dim = action_space.shape[-1]

        cfg["temp_target_entropy"] = cfg["temp_target_entropy_coef"] * self._action_dim

        super(SimbaAgent, self).__init__(
            observation_space,
            action_space,
            cfg,
        )

        # map dictionary to dataclass
        self._cfg = SimbaConfig(**cfg)

        # initialize networks
        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
        ) = _init_simba_networks(self._observation_dim, self._action_dim, self._cfg)

    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool,
    ) -> np.ndarray:
        if training:
            temperature = 1.0
        else:
            temperature = 0.0

        # current timestep observation is "next" observations from the previous timestep
        observations = jnp.asarray(prev_timestep["next_observation"])

        self._rng, actions = _sample_simba_actions(
            self._rng, self._actor, observations, temperature
        )
        actions = np.array(actions)

        return actions

    def update(self, update_step: int, batch: Dict[str, np.ndarray]) -> Dict:
        for key, value in batch.items():
            batch[key] = jnp.asarray(value)

        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
            update_info,
        ) = _update_simba_networks(
            rng=self._rng,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            batch=batch,
            gamma=self._cfg.gamma,
            n_step=self._cfg.n_step,
            actor_bc_alpha=self._cfg.actor_bc_alpha,
            critic_use_cdq=self._cfg.critic_use_cdq,
            target_tau=self._cfg.target_tau,
            temp_target_entropy=self._cfg.temp_target_entropy,
        )

        for key, value in update_info.items():
            if isinstance(value, dict):
                continue
            update_info[key] = float(value)

        return update_info

    def save(self, path: str) -> None:
        self._actor.save(path + "/actor")
        self._critic.save(path + "/critic")
        self._target_critic.save(path + "/target_critic")
        self._temperature.save(path + "/temperature")

    def load(self, path: str) -> None:
        only_param = self._cfg.load_only_param
        param_key = self._cfg.load_param_key

        self._actor = self._actor.load(path + "/actor", param_key, only_param)
        self._critic = self._critic.load(path + "/critic", param_key, only_param)
        self._target_critic = self._target_critic.load(
            path + "/target_critic", param_key, only_param
        )
        self._temperature = self._temperature.load(
            path + "/temperature", None, only_param
        )
