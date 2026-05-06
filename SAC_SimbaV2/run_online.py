import argparse
import math
import os

# Set XLA flags to enforce deterministic operations on GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import random
import sys

import hydra
import numpy as np
import omegaconf
import tqdm
from dotmap import DotMap

from scale_rl.agents import create_agent
from scale_rl.buffers import create_buffer
from scale_rl.common import WandbTrainerLogger
from scale_rl.envs import create_envs
from scale_rl.evaluation import evaluate, record_video


def _is_lora_enabled(agent_cfg) -> bool:
    """Check if any LoRA component is enabled in the agent config."""
    for key in ['actor_low_rank', 'actor_encoder_low_rank', 'actor_head_low_rank',
                'critic_low_rank', 'critic_encoder_low_rank', 'critic_head_low_rank']:
        try:
            sub = getattr(agent_cfg, key, None)
            if sub is not None and getattr(sub, 'enable', False):
                return True
        except (AttributeError, TypeError):
            pass
    return False


def run(args):
    ###############################
    # configs
    ###############################

    args = DotMap(args)
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides

    hydra.initialize(version_base=None, config_path=config_path)
    cfg = hydra.compose(config_name=config_name, overrides=overrides)

    def eval_resolver(s: str):
        return eval(s)

    omegaconf.OmegaConf.register_new_resolver("eval", eval_resolver)
    omegaconf.OmegaConf.resolve(cfg)

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    #############################
    # envs
    #############################
    train_env, eval_env = create_envs(**cfg.env)

    observation_space = train_env.observation_space
    action_space = train_env.action_space

    #############################
    # buffer
    #############################
    buffer = create_buffer(
        observation_space=observation_space, action_space=action_space, **cfg.buffer
    )
    buffer.reset()

    #############################
    # agent
    #############################

    # Since the network architecture is typically tied to the learning algorithm,
    #   we opted not to fully modularize the network for the sake of readability.
    # Therefore, for each algorithm, the network is implemented within its respective directory.

    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        cfg=cfg.agent,
    )

    #############################
    # train
    #############################

    # Determine wandb directory based on LoRA status (absolute path required by orbax)
    wandb_dir = "wandb_lora" if _is_lora_enabled(cfg.agent) else "wandb_base"
    wandb_dir = os.path.abspath(wandb_dir)

    run_metadata = None
    if hasattr(agent, "get_logging_metadata"):
        run_metadata = agent.get_logging_metadata()
        run_metadata.update(
            {
                "algo": "sac",
                "env_name": cfg.env.env_name,
                "seed": cfg.seed,
            }
        )
    logger = WandbTrainerLogger(cfg, run_metadata=run_metadata, wandb_dir=wandb_dir)

    # load model if given
    if cfg.load_path:
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        load_path = script_dir + "/" + cfg.load_path
        agent.load(load_path)

    # save initial model if save_model is enabled
    save_model = getattr(cfg, "save_model", False)
    if save_model and logger.run and getattr(logger.run, "dir", None):
        init_path = os.path.join(logger.run.dir, "init_model")
        os.makedirs(init_path, exist_ok=True)
        agent.save(init_path)

    # initial evaluation
    eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes)
    logger.update_metric(**eval_info)
    logger.log_metric(step=0)
    logger.reset()

    # start training
    update_step = 0
    update_counter = 0
    observations, env_infos = train_env.reset()
    timestep = None
    media_dir = None
    if logger.run and getattr(logger.run, "dir", None):
        media_dir = os.path.join(logger.run.dir, "media")
        os.makedirs(media_dir, exist_ok=True)

    for interaction_step in tqdm.tqdm(
        range(1, int(cfg.num_interaction_steps + 1)), smoothing=0.1
    ):
        # collect data
        # While using random actions until buffer.can_sample(),
        # we feed data into agent to compute statistics within a wrapper.
        if timestep:
            actions = agent.sample_actions(
                interaction_step, prev_timestep=timestep, training=True
            )
        if buffer.can_sample() is False:
            actions = train_env.action_space.sample()

        next_observations, rewards, terminateds, truncateds, env_infos = train_env.step(
            actions
        )
        next_buffer_observations = next_observations.copy()
        for env_idx in range(cfg.num_train_envs):
            if terminateds[env_idx] or truncateds[env_idx]:
                next_buffer_observations[env_idx] = env_infos["final_obs"][env_idx]

        timestep = {
            "observation": observations,
            "action": actions,
            "reward": rewards,
            "terminated": terminateds,
            "truncated": truncateds,
            "next_observation": next_buffer_observations,
        }
        buffer.add(timestep)
        timestep["next_observation"] = next_observations
        observations = next_observations

        if buffer.can_sample():
            # update network
            # updates_per_interaction_step can be below 1.0
            update_counter += cfg.updates_per_interaction_step
            while update_counter >= 1:
                batch = buffer.sample()
                update_info = agent.update(update_step, batch)
                logger.update_metric(**update_info)
                update_counter -= 1
                update_step += 1

            # evaluation
            if interaction_step % cfg.evaluation_per_interaction_step == 0:
                eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes)
                logger.update_metric(**eval_info)

            # metrics
            if interaction_step % cfg.metrics_per_interaction_step == 0:
                batch = buffer.sample()
                metrics_info = agent.get_metrics(batch, update_info)
                if metrics_info:
                    logger.update_metric(**metrics_info)

            # video recording
            if interaction_step % cfg.recording_per_interaction_step == 0 and cfg.num_record_episodes > 0:
                video_info = record_video(
                    agent,
                    eval_env,
                    cfg.num_record_episodes,
                    media_dir=media_dir,
                    prefix=f"step_{interaction_step}",
                )
                logger.update_metric(**video_info)

            # logging
            if interaction_step % cfg.logging_per_interaction_step == 0:
                # using env steps simplifies the comparison with the performance reported in the paper.
                env_step = interaction_step * cfg.action_repeat * cfg.num_train_envs
                logger.log_metric(step=env_step)
                logger.reset()

    # save final model if save_model is enabled
    if save_model and logger.run and getattr(logger.run, "dir", None):
        final_path = os.path.join(logger.run.dir, "final_model")
        os.makedirs(final_path, exist_ok=True)
        agent.save(final_path)
        if media_dir is not None and cfg.num_record_episodes > 0:
            record_video(
                agent,
                eval_env,
                cfg.num_record_episodes,
                media_dir=media_dir,
                prefix="final",
            )

    train_env.close()
    eval_env.close()
    logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="online_rl")
    parser.add_argument("--overrides", action="append", default=[])
    args = parser.parse_args()

    run(vars(args))
