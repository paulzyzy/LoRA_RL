import os
from datetime import datetime
from typing import Dict, Optional

import gymnasium as gym
import imageio
import numpy as np
from gymnasium.vector import VectorEnv

import wandb


def evaluate(
    agent,
    env: VectorEnv,
    num_episodes: int,
) -> Dict[str, float]:
    n = env.num_envs

    assert num_episodes % n == 0, "num_episodes must be divisible by env.num_envs"
    num_eval_episodes_per_env = num_episodes // n

    total_returns = []
    total_successes = []
    total_lengths = []

    for _ in range(num_eval_episodes_per_env):
        returns = np.zeros(n)
        lengths = np.zeros(n)
        successes = np.zeros(n)

        observations, infos = env.reset()

        prev_timestep = {"next_observation": observations}

        dones = np.zeros(n)
        while np.sum(dones) < n:
            actions = agent.sample_actions(
                interaction_step=0,
                prev_timestep=prev_timestep,
                training=False,
            )
            next_observations, rewards, terminateds, truncateds, infos = env.step(
                actions
            )

            prev_timestep = {"next_observation": next_observations}

            returns += rewards * (1 - dones)
            lengths += 1 - dones

            if "success" in infos:
                successes += infos["success"].astype("float") * (1 - dones)

            elif "final_info" in infos:
                final_successes = np.zeros(n)
                for idx in range(n):
                    final_info = infos["final_info"]

                    if "success" in final_info:
                        try:
                            final_successes[idx] = final_info["success"][idx].astype(
                                "float"
                            )
                        except:
                            final_successes[idx] = np.array(
                                final_info["success"][idx]
                            ).astype("float")
                successes += final_successes * (1 - dones)

            else:
                pass

            # once an episode is done in a sub-environment, we assume it to be done.
            # also, we assume to be done whether it is terminated or truncated during evaluation.
            dones = np.maximum(dones, terminateds)
            dones = np.maximum(dones, truncateds)

            # proceed
            observations = next_observations

        for env_idx in range(n):
            total_returns.append(returns[env_idx])
            total_lengths.append(lengths[env_idx])
            total_successes.append(successes[env_idx].astype("bool").astype("float"))

    eval_info = {
        "avg_return": np.mean(total_returns),
        "avg_length": np.mean(total_lengths),
        "avg_success": np.mean(total_successes),
    }

    return eval_info


def record_video(
    agent,
    env: VectorEnv,
    num_episodes: int,
    video_length: int = 100,
    media_dir: Optional[str] = None,
    prefix: Optional[str] = None,
) -> Dict[str, float]:
    if num_episodes <= 0:
        return {}

    n = env.num_envs
    assert num_episodes % n == 0, "num_episodes must be divisible by env.num_envs"
    num_eval_episodes_per_env = num_episodes // n

    total_videos = []

    for _ in range(num_eval_episodes_per_env):
        videos = []

        observations, infos = env.reset()
        prev_timestep = {"next_observation": observations}
        images = env.call("render")
        dones = np.zeros(n)
        while np.sum(dones) < n:
            actions = agent.sample_actions(
                interaction_step=0,
                prev_timestep=prev_timestep,
                training=False,
            )
            next_observations, rewards, terminateds, truncateds, infos = env.step(
                actions
            )

            prev_timestep = {"next_observation": next_observations}

            # once an episode is done in a sub-environment, we assume it to be done.
            dones = np.maximum(dones, terminateds)
            dones = np.maximum(dones, truncateds)

            # proceed
            videos.append(images)
            images = env.call("render")
            observations = next_observations

        total_videos.append(np.stack(videos, axis=1))  # (n, t, c, h, w)

    total_videos = np.concatenate(total_videos, axis=0)  # (b, t, h, w, c)
    total_videos = total_videos[:, :video_length]
    total_videos = total_videos.transpose(0, 1, 4, 2, 3)  # (b, t, c, h, w)

    if media_dir:
        os.makedirs(media_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = prefix or "episode"
        for vid_idx, clip in enumerate(total_videos):
            frames = np.transpose(clip, (0, 2, 3, 1))  # (t, h, w, c)
            frames = np.clip(frames, 0, 255).astype(np.uint8)
            filename = os.path.join(
                media_dir, f"{base_name}_{timestamp}_{vid_idx}.mp4"
            )
            try:
                imageio.mimwrite(filename, frames, fps=10, codec="libx264", quality=8)
            except Exception:
                fallback = filename.replace(".mp4", ".gif")
                imageio.mimsave(fallback, frames, fps=10)

    video_info = {"video": wandb.Video(total_videos, fps=10, format="gif")}

    return video_info
