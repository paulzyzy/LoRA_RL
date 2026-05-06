import os
import sys

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import random
import time
import math

import tqdm
import wandb
import numpy as np

try:
    # Required for avoiding IsaacGym import error
    import isaacgym
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

from tensordict import TensorDict

from fast_td3_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    PerTaskRewardNormalizer,
    SimpleReplayBuffer,
    save_params,
    mark_step,
)
from hyperparams import get_args
from lora import (
    replace_linear_with_lora,
    scale_lora_base_weights,
    scale_lora_AB,
    get_lora_params,
    get_non_lora_trainable_params,
    soft_update_with_lora,
)

torch.set_float32_matmul_precision("high")

try:
    import jax.numpy as jnp
except ImportError:
    pass


def _count_params(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _format_params(n):
    """Format parameter count as human-readable string."""
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


@torch.no_grad()
def _compute_rank_metrics(model: nn.Module, obs: torch.Tensor, prefix: str, is_critic: bool = False, actions: torch.Tensor = None):
    """Compute srank and effective_rank averaged across encoder residual block layers.

    For SimbaV2 architectures, hooks are registered on Linear layers inside
    the encoder blocks to capture activations.  The singular values of each
    activation matrix are then used to compute:
      - srank: number of singular values needed to reach 99% of the total sum
      - effective_rank: number of squared singular values for 95% of variance

    Returns a dict of metric_name -> float.
    """
    from fast_td3_simbav2 import HyperLERPBlock

    activations = []
    hooks = []

    def _make_hook(storage):
        def hook_fn(module, input, output):
            storage.append(output.detach().float())
        return hook_fn

    # Collect the target modules to hook into
    if is_critic:
        # Hook into encoder blocks of qnet1 only (qnet2 is symmetric)
        encoder_modules = [model.qnet1]
    else:
        encoder_modules = [model]

    for top_mod in encoder_modules:
        if not hasattr(top_mod, 'encoder'):
            continue
        for block in top_mod.encoder.modules():
            if isinstance(block, HyperLERPBlock):
                # Hook the output of each residual block
                h = block.register_forward_hook(_make_hook(activations))
                hooks.append(h)

    if not hooks:
        return {}

    # Forward pass to collect activations
    model.eval()
    if is_critic and actions is not None:
        model(obs, actions)
    else:
        model(obs)
    model.train()

    # Remove hooks
    for h in hooks:
        h.remove()

    if not activations:
        return {}

    srank_vals = []
    erank_vals = []
    for feat in activations:
        if len(feat.shape) > 2:
            feat = feat.reshape(-1, feat.shape[-1])
        svals = torch.linalg.svdvals(feat)
        sval_sum = svals.sum()
        cumsum = torch.cumsum(svals, dim=0)
        # srank: 99% of singular value sum
        srank_crossed = cumsum >= 0.99 * sval_sum
        srank_vals.append((~srank_crossed).sum().float() + 1)
        # effective rank: 95% of squared singular value sum (variance)
        sval_sq = svals ** 2
        sval_sq_sum = sval_sq.sum()
        cumsum_sq = torch.cumsum(sval_sq, dim=0)
        erank_crossed = cumsum_sq >= 0.95 * sval_sq_sum
        erank_vals.append((~erank_crossed).sum().float() + 1)

    metrics = {
        f"{prefix}/srank_mean": torch.stack(srank_vals).mean().item(),
        f"{prefix}/effective_rank_mean": torch.stack(erank_vals).mean().item(),
    }
    return metrics


def main():
    args = get_args()
    print(args)

    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "mps" if args.cuda and torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{args.device_rank}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    if args.env_name.startswith("h1hand-") or args.env_name.startswith("h1-"):
        from environments.humanoid_bench_env import HumanoidBenchEnv

        env_type = "humanoid_bench"
        envs = HumanoidBenchEnv(args.env_name, args.num_envs, device=device)
        eval_envs = envs
        render_env = HumanoidBenchEnv(
            args.env_name, 1, render_mode="rgb_array", device=device
        )
    elif args.env_name.startswith("Isaac-"):
        from environments.isaaclab_env import IsaacLabEnv

        env_type = "isaaclab"
        envs = IsaacLabEnv(
            args.env_name,
            device.type,
            args.num_envs,
            args.seed,
            action_bounds=args.action_bounds,
        )
        eval_envs = envs
        render_env = envs
    elif args.env_name.startswith("MTBench-"):
        from environments.mtbench_env import MTBenchEnv

        env_name = "-".join(args.env_name.split("-")[1:])
        env_type = "mtbench"
        envs = MTBenchEnv(env_name, args.device_rank, args.num_envs, args.seed)
        eval_envs = envs
        render_env = envs
    else:
        from environments.mujoco_playground_env import make_env

        # TODO: Check if re-using same envs for eval could reduce memory usage
        env_type = "mujoco_playground"
        envs, eval_envs, render_env = make_env(
            args.env_name,
            args.seed,
            args.num_envs,
            args.num_eval_envs,
            args.device_rank,
            use_tuned_reward=args.use_tuned_reward,
            use_domain_randomization=args.use_domain_randomization,
            use_push_randomization=args.use_push_randomization,
        )

    n_act = envs.num_actions
    n_obs = envs.num_obs if type(envs.num_obs) == int else envs.num_obs[0]
    if envs.asymmetric_obs:
        n_critic_obs = (
            envs.num_privileged_obs
            if type(envs.num_privileged_obs) == int
            else envs.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs
    action_low, action_high = -1.0, 1.0

    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(
            shape=n_critic_obs, device=device
        )
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()

    if args.reward_normalization:
        if env_type in ["mtbench"]:
            reward_normalizer = PerTaskRewardNormalizer(
                num_tasks=envs.num_tasks,
                gamma=args.gamma,
                device=device,
                g_max=min(abs(args.v_min), abs(args.v_max)),
            )
        else:
            reward_normalizer = RewardNormalizer(
                gamma=args.gamma,
                device=device,
                g_max=min(abs(args.v_min), abs(args.v_max)),
            )
    else:
        reward_normalizer = nn.Identity()

    actor_kwargs = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_envs": args.num_envs,
        "device": device,
        "init_scale": args.init_scale,
        "hidden_dim": args.actor_hidden_dim,
        "std_min": args.std_min,
        "std_max": args.std_max,
    }
    critic_kwargs = {
        "n_obs": n_critic_obs,
        "n_act": n_act,
        "num_atoms": args.num_atoms,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "hidden_dim": args.critic_hidden_dim,
        "device": device,
    }

    if env_type == "mtbench":
        actor_kwargs["n_obs"] = n_obs - envs.num_tasks + args.task_embedding_dim
        critic_kwargs["n_obs"] = n_critic_obs - envs.num_tasks + args.task_embedding_dim
        actor_kwargs["num_tasks"] = envs.num_tasks
        actor_kwargs["task_embedding_dim"] = args.task_embedding_dim
        critic_kwargs["num_tasks"] = envs.num_tasks
        critic_kwargs["task_embedding_dim"] = args.task_embedding_dim

    if args.agent == "fasttd3":
        if env_type in ["mtbench"]:
            from fast_td3 import MultiTaskActor, MultiTaskCritic

            actor_cls = MultiTaskActor
            critic_cls = MultiTaskCritic
        else:
            from fast_td3 import Actor, Critic

            actor_cls = Actor
            critic_cls = Critic

        print("Using FastTD3")
    elif args.agent == "fasttd3_simbav2":
        if env_type in ["mtbench"]:
            from fast_td3_simbav2 import MultiTaskActor, MultiTaskCritic

            actor_cls = MultiTaskActor
            critic_cls = MultiTaskCritic
        else:
            from fast_td3_simbav2 import Actor, Critic

            actor_cls = Actor
            critic_cls = Critic

        print("Using FastTD3 + SimbaV2")
        actor_kwargs.pop("init_scale")
        actor_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.actor_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.actor_hidden_dim),
                "alpha_init": 1.0 / (args.actor_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.actor_hidden_dim),
                "expansion": 4,
                "c_shift": 3.0,
                "num_blocks": args.actor_num_blocks,
            }
        )
        critic_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.critic_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.critic_hidden_dim),
                "alpha_init": 1.0 / (args.critic_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.critic_hidden_dim),
                "num_blocks": args.critic_num_blocks,
                "expansion": 4,
                "c_shift": 3.0,
            }
        )
    else:
        raise ValueError(f"Agent {args.agent} not supported")

    actor = actor_cls(**actor_kwargs)

    # --- LoRA setup for actor ---
    actor_has_lora = args.actor_lora_enable
    if actor_has_lora:
        if args.agent != "fasttd3_simbav2":
            raise ValueError("LoRA is only supported for fasttd3_simbav2 residual blocks.")
        print(f"Applying LoRA to actor residual blocks: rank={args.lora_rank}, alpha={args.lora_alpha}")
        replace_linear_with_lora(
            actor.encoder, args.lora_rank, args.lora_alpha,
            args.lora_a_init, args.lora_b_init, device,
        )
        scale_lora_base_weights(actor.encoder, args.kernel_init_scale)
        scale_lora_AB(actor.encoder, args.kernel_init_scale)

    if env_type in ["mtbench"]:
        # Python 3.8 doesn't support 'from_module' in tensordict
        policy = actor.explore
    else:
        from tensordict import from_module

        actor_detach = actor_cls(**actor_kwargs)
        if actor_has_lora:
            replace_linear_with_lora(
                actor_detach.encoder, args.lora_rank, args.lora_alpha,
                args.lora_a_init, args.lora_b_init, device,
            )
        # Copy params to actor_detach without grad
        from_module(actor).data.to_module(actor_detach)
        policy = actor_detach.explore

    qnet = critic_cls(**critic_kwargs)

    # --- LoRA setup for critic ---
    critic_has_lora = args.critic_lora_enable
    if critic_has_lora:
        if args.agent != "fasttd3_simbav2":
            raise ValueError("LoRA is only supported for fasttd3_simbav2 residual blocks.")
        print(f"Applying LoRA to critic residual blocks: rank={args.lora_rank}, alpha={args.lora_alpha}")
        for qnet_sub in [qnet.qnet1, qnet.qnet2]:
            replace_linear_with_lora(
                qnet_sub.encoder, args.lora_rank, args.lora_alpha,
                args.lora_a_init, args.lora_b_init, device,
            )
            scale_lora_base_weights(qnet_sub.encoder, args.kernel_init_scale)
            scale_lora_AB(qnet_sub.encoder, args.kernel_init_scale)

    qnet_target = critic_cls(**critic_kwargs)
    if critic_has_lora:
        for qnet_sub in [qnet_target.qnet1, qnet_target.qnet2]:
            replace_linear_with_lora(
                qnet_sub.encoder, args.lora_rank, args.lora_alpha,
                args.lora_a_init, args.lora_b_init, device,
            )
    qnet_target.load_state_dict(qnet.state_dict())

    # --- Parameter counting and run name ---
    actor_total, actor_trainable = _count_params(actor)
    critic_total, critic_trainable = _count_params(qnet)

    print(f"[Actor]  total params={_format_params(actor_total)}, trainable params={_format_params(actor_trainable)}")
    print(f"[Critic] total params={_format_params(critic_total)}, trainable params={_format_params(critic_trainable)}")

    # Build informative run name
    run_name_parts = [args.env_name, str(args.seed)]
    run_name_parts.append(f"AP={_format_params(actor_total)}")
    run_name_parts.append(f"CP={_format_params(critic_total)}")
    if actor_has_lora or critic_has_lora:
        run_name_parts.append(f"Tr_A={_format_params(actor_trainable)}")
        run_name_parts.append(f"Tr_C={_format_params(critic_trainable)}")
    if critic_has_lora:
        run_name_parts.append(f"CriLowR=r{args.lora_rank}")
    if actor_has_lora:
        run_name_parts.append(f"ActLowR=r{args.lora_rank}")
    if critic_has_lora or actor_has_lora:
        run_name_parts.append(f"alpha={args.lora_alpha}")
        run_name_parts.append(f"lora_lr={args.lora_learning_rate}")
        run_name_parts.append(f"lora_wd={args.lora_weight_decay}")
    if args.agent == "fasttd3_simbav2":
        run_name_parts.append(f"AW{args.actor_hidden_dim}_AD{args.actor_num_blocks}")
        run_name_parts.append(f"CW{args.critic_hidden_dim}_CD{args.critic_num_blocks}")
    run_name = "_".join(run_name_parts)
    print(f"Run name: {run_name}")

    if args.use_wandb:
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    # --- Optimizer setup ---
    if actor_has_lora:
        # LoRA: separate param groups for LoRA params and base trainable params (biases)
        actor_lora_ps = get_lora_params(actor)
        actor_base_ps = get_non_lora_trainable_params(actor)
        actor_param_groups = [
            {
                "params": actor_lora_ps,
                "lr": torch.tensor(args.lora_learning_rate, device=device),
                "weight_decay": args.lora_weight_decay,
            },
        ]
        if actor_base_ps:
            actor_param_groups.append(
                {
                    "params": actor_base_ps,
                    "lr": torch.tensor(args.actor_learning_rate, device=device),
                    "weight_decay": args.weight_decay,
                },
            )
        actor_optimizer = optim.AdamW(actor_param_groups)
    else:
        actor_optimizer = optim.AdamW(
            list(actor.parameters()),
            lr=torch.tensor(args.actor_learning_rate, device=device),
            weight_decay=args.weight_decay,
        )

    if critic_has_lora:
        critic_lora_ps = get_lora_params(qnet)
        critic_base_ps = get_non_lora_trainable_params(qnet)
        critic_param_groups = [
            {
                "params": critic_lora_ps,
                "lr": torch.tensor(args.lora_learning_rate, device=device),
                "weight_decay": args.lora_weight_decay,
            },
        ]
        if critic_base_ps:
            critic_param_groups.append(
                {
                    "params": critic_base_ps,
                    "lr": torch.tensor(args.critic_learning_rate, device=device),
                    "weight_decay": args.weight_decay,
                },
            )
        q_optimizer = optim.AdamW(critic_param_groups)
    else:
        q_optimizer = optim.AdamW(
            list(qnet.parameters()),
            lr=torch.tensor(args.critic_learning_rate, device=device),
            weight_decay=args.weight_decay,
        )

    # Add learning rate schedulers
    def _cosine_lambda(lr_init, lr_end, total_steps):
        """Per-group cosine decay: lr_init -> lr_end over total_steps."""
        def fn(step):
            if lr_init == 0:
                return 1.0
            ratio = lr_end / lr_init
            return ratio + (1 - ratio) * (1 + math.cos(math.pi * step / total_steps)) / 2
        return fn

    if critic_has_lora:
        # Per-group schedule: LoRA params -> lora_lr_end, base params -> critic_lr_end
        q_lambdas = [_cosine_lambda(args.lora_learning_rate, args.lora_learning_rate_end, args.total_timesteps)]
        if critic_base_ps:
            q_lambdas.append(_cosine_lambda(args.critic_learning_rate, args.critic_learning_rate_end, args.total_timesteps))
        q_scheduler = optim.lr_scheduler.LambdaLR(q_optimizer, q_lambdas)
    else:
        q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            q_optimizer,
            T_max=args.total_timesteps,
            eta_min=torch.tensor(args.critic_learning_rate_end, device=device),
        )

    if actor_has_lora:
        actor_lambdas = [_cosine_lambda(args.lora_learning_rate, args.lora_learning_rate_end, args.total_timesteps)]
        if actor_base_ps:
            actor_lambdas.append(_cosine_lambda(args.actor_learning_rate, args.actor_learning_rate_end, args.total_timesteps))
        actor_scheduler = optim.lr_scheduler.LambdaLR(actor_optimizer, actor_lambdas)
    else:
        actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            actor_optimizer,
            T_max=args.total_timesteps,
            eta_min=torch.tensor(args.actor_learning_rate_end, device=device),
        )

    rb = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        asymmetric_obs=envs.asymmetric_obs,
        playground_mode=env_type == "mujoco_playground",
        n_steps=args.num_steps,
        gamma=args.gamma,
        device=device,
    )

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip

    def evaluate():
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=device)
        episode_lengths = torch.zeros(num_eval_envs, device=device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

        if env_type == "isaaclab":
            obs = eval_envs.reset(random_start_init=False)
        else:
            obs = eval_envs.reset()

        # Run for a fixed number of steps
        for i in range(eval_envs.max_episode_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                obs = normalize_obs(obs, update=False)
                actions = actor(obs)

            next_obs, rewards, dones, infos = eval_envs.step(actions.float())

            if env_type == "mtbench":
                # We only report success rate in MTBench evaluation
                rewards = (
                    infos["episode"]["success"].float() if "episode" in infos else 0.0
                )
            episode_returns = torch.where(
                ~done_masks, episode_returns + rewards, episode_returns
            )
            episode_lengths = torch.where(
                ~done_masks, episode_lengths + 1, episode_lengths
            )
            if env_type == "mtbench" and "episode" in infos:
                dones = dones | infos["episode"]["success"]
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs

        return episode_returns.mean().item(), episode_lengths.mean().item()

    def render_with_rollout():
        # Quick rollout for rendering
        if env_type == "humanoid_bench":
            obs = render_env.reset()
            renders = [render_env.render()]
        elif env_type in ["isaaclab", "mtbench"]:
            raise NotImplementedError(
                "We don't support rendering for IsaacLab and MTBench environments"
            )
        else:
            obs = render_env.reset()
            render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            renders = [render_env.state]
        for i in range(render_env.max_episode_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                obs = normalize_obs(obs, update=False)
                actions = actor(obs)
            next_obs, _, done, _ = render_env.step(actions.float())
            if env_type == "mujoco_playground":
                render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            if i % 2 == 0:
                if env_type == "humanoid_bench":
                    renders.append(render_env.render())
                else:
                    renders.append(render_env.state)
            if done.any():
                break
            obs = next_obs

        if env_type == "mujoco_playground":
            renders = render_env.render_trajectory(renders)
        return renders

    def update_main(data, logs_dict):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            if envs.asymmetric_obs:
                critic_observations = data["critic_observations"]
                next_critic_observations = data["next"]["critic_observations"]
            else:
                critic_observations = observations
                next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            if args.disable_bootstrap:
                bootstrap = (~dones).float()
            else:
                bootstrap = (truncations | ~dones).float()

            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(policy_noise).clamp(
                -noise_clip, noise_clip
            )

            next_state_actions = (actor(next_observations) + clipped_noise).clamp(
                action_low, action_high
            )
            discount = args.gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
                qf1_next_target_projected, qf2_next_target_projected = (
                    qnet_target.projection(
                        next_critic_observations,
                        next_state_actions,
                        rewards,
                        bootstrap,
                        discount,
                    )
                )
                qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
                qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)
                if args.use_cdq:
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1)
                        < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
                else:
                    qf1_next_target_dist, qf2_next_target_dist = (
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )

            qf1, qf2 = qnet(critic_observations, actions)
            qf1_loss = -torch.sum(
                qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
            ).mean()
            qf2_loss = -torch.sum(
                qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
            ).mean()
            qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if args.use_grad_norm_clipping:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(q_optimizer)
        scaler.update()

        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = qf1_next_target_value.max().detach()
        logs_dict["qf_min"] = qf1_next_target_value.min().detach()
        return logs_dict

    def update_pol(data, logs_dict):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            critic_observations = (
                data["critic_observations"]
                if envs.asymmetric_obs
                else data["observations"]
            )

            qf1, qf2 = qnet(critic_observations, actor(data["observations"]))
            qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
            qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
            if args.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        if args.use_grad_norm_clipping:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(actor_optimizer)
        scaler.update()

        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["actor_loss"] = actor_loss.detach()
        return logs_dict

    @torch.no_grad()
    def soft_update(src, tgt, tau: float):
        if critic_has_lora:
            # LoRA-aware soft update: merge effective weights, zero target B
            soft_update_with_lora(src, tgt, tau)
        else:
            src_ps = [p.data for p in src.parameters()]
            tgt_ps = [p.data for p in tgt.parameters()]
            torch._foreach_mul_(tgt_ps, 1.0 - tau)
            torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    if args.compile:
        compile_mode = args.compile_mode
        update_main = torch.compile(update_main, mode=compile_mode)
        update_pol = torch.compile(update_pol, mode=compile_mode)
        policy = torch.compile(policy, mode=None)
        normalize_obs = torch.compile(obs_normalizer.forward, mode=None)
        normalize_critic_obs = torch.compile(critic_obs_normalizer.forward, mode=None)
        if args.reward_normalization:
            update_stats = torch.compile(reward_normalizer.update_stats, mode=None)
        normalize_reward = torch.compile(reward_normalizer.forward, mode=None)
    else:
        normalize_obs = obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward
        if args.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    if envs.asymmetric_obs:
        obs, critic_obs = envs.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs = envs.reset()
    if args.checkpoint_path:
        # Load checkpoint if specified
        torch_checkpoint = torch.load(
            f"{args.checkpoint_path}", map_location=device, weights_only=False
        )
        actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        critic_obs_normalizer.load_state_dict(
            torch_checkpoint["critic_obs_normalizer_state"]
        )
        qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        global_step = torch_checkpoint["global_step"]
    else:
        global_step = 0

    dones = None
    pbar = tqdm.tqdm(total=args.total_timesteps, initial=global_step)
    start_time = None
    desc = ""

    while global_step < args.total_timesteps:
        mark_step()
        logs_dict = TensorDict()
        if (
            start_time is None
            and global_step >= args.measure_burnin + args.learning_starts
        ):
            start_time = time.time()
            measure_burnin = global_step

        with torch.no_grad(), autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            norm_obs = normalize_obs(obs)
            actions = policy(obs=norm_obs, dones=dones)

        next_obs, rewards, dones, infos = envs.step(actions.float())
        truncations = infos["time_outs"]

        if args.reward_normalization:
            if env_type == "mtbench":
                task_ids_one_hot = obs[..., -envs.num_tasks :]
                task_indices = torch.argmax(task_ids_one_hot, dim=1)
                update_stats(rewards, dones.float(), task_ids=task_indices)
            else:
                update_stats(rewards, dones.float())

        if envs.asymmetric_obs:
            next_critic_obs = infos["observations"]["critic"]
        # Compute 'true' next_obs and next_critic_obs for saving
        true_next_obs = torch.where(
            dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
        )
        if envs.asymmetric_obs:
            true_next_critic_obs = torch.where(
                dones[:, None] > 0,
                infos["observations"]["raw"]["critic_obs"],
                next_critic_obs,
            )

        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "rewards": torch.as_tensor(
                        rewards, device=device, dtype=torch.float
                    ),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )
        if envs.asymmetric_obs:
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = true_next_critic_obs
        rb.extend(transition)

        obs = next_obs
        if envs.asymmetric_obs:
            critic_obs = next_critic_obs

        if global_step > args.learning_starts and global_step % args.update_interval == 0:
            for i in range(args.num_updates):
                data = rb.sample(max(1, args.batch_size // args.num_envs))
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(
                    data["next"]["observations"]
                )
                if envs.asymmetric_obs:
                    data["critic_observations"] = normalize_critic_obs(
                        data["critic_observations"]
                    )
                    data["next"]["critic_observations"] = normalize_critic_obs(
                        data["next"]["critic_observations"]
                    )
                raw_rewards = data["next"]["rewards"]
                if env_type in ["mtbench"] and args.reward_normalization:
                    # Multi-task reward normalization
                    task_ids_one_hot = data["observations"][..., -envs.num_tasks :]
                    task_indices = torch.argmax(task_ids_one_hot, dim=1)
                    data["next"]["rewards"] = normalize_reward(
                        raw_rewards, task_ids=task_indices
                    )
                else:
                    data["next"]["rewards"] = normalize_reward(raw_rewards)

                logs_dict = update_main(data, logs_dict)
                if args.num_updates > 1:
                    if i % args.policy_frequency == 1:
                        logs_dict = update_pol(data, logs_dict)
                else:
                    if global_step % args.policy_frequency == 0:
                        logs_dict = update_pol(data, logs_dict)

                soft_update(qnet, qnet_target, args.tau)

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "actor_loss": logs_dict["actor_loss"].mean(),
                        "qf_loss": logs_dict["qf_loss"].mean(),
                        "qf_max": logs_dict["qf_max"].mean(),
                        "qf_min": logs_dict["qf_min"].mean(),
                        "actor_grad_norm": logs_dict["actor_grad_norm"].mean(),
                        "critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                        "env_rewards": rewards.mean(),
                        "buffer_rewards": raw_rewards.mean(),
                    }

                    # Compute rank metrics at eval intervals (SVD is expensive)
                    # Works for any SimbaV2 run (with or without LoRA) so baselines can be compared
                    if args.agent == "fasttd3_simbav2" and args.eval_interval > 0 and global_step % args.eval_interval == 0:
                        rank_obs = data["observations"][:256]  # use small batch for efficiency
                        actor_rank_metrics = _compute_rank_metrics(actor, rank_obs, "actor", is_critic=False)
                        logs.update(actor_rank_metrics)
                        rank_actions = data["actions"][:256]
                        critic_rank_obs = data["critic_observations"][:256] if envs.asymmetric_obs else rank_obs
                        critic_rank_metrics = _compute_rank_metrics(qnet, critic_rank_obs, "critic", is_critic=True, actions=rank_actions)
                        logs.update(critic_rank_metrics)

                    if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                        print(f"Evaluating at global step {global_step}")
                        eval_avg_return, eval_avg_length = evaluate()
                        if env_type in ["humanoid_bench", "isaaclab", "mtbench"]:
                            # NOTE: Hacky way of evaluating performance, but just works
                            obs = envs.reset()
                        logs["eval_avg_return"] = eval_avg_return
                        logs["eval_avg_length"] = eval_avg_length

                    if (
                        args.render_interval > 0
                        and global_step % args.render_interval == 0
                    ):
                        renders = render_with_rollout()
                        render_video = wandb.Video(
                            np.array(renders).transpose(
                                0, 3, 1, 2
                            ),  # Convert to (T, C, H, W) format
                            fps=30,
                            format="gif",
                        )
                        logs["render_video"] = render_video
                if args.use_wandb:
                    wandb.log(
                        {
                            "speed": speed,
                            "frame": global_step * args.num_envs,
                            "critic_lr": q_scheduler.get_last_lr()[0],
                            "actor_lr": actor_scheduler.get_last_lr()[0],
                            **logs,
                        },
                        step=global_step,
                    )

            if (
                args.save_interval > 0
                and global_step > 0
                and global_step % args.save_interval == 0
            ):
                print(f"Saving model at global step {global_step}")
                save_params(
                    global_step,
                    actor,
                    qnet,
                    qnet_target,
                    obs_normalizer,
                    critic_obs_normalizer,
                    args,
                    f"models/{run_name}_{global_step}.pt",
                )

        global_step += 1
        actor_scheduler.step()
        q_scheduler.step()
        pbar.update(1)

    # Final evaluation
    if args.eval_interval > 0:
        print(f"Final evaluation at global step {global_step}")
        eval_avg_return, eval_avg_length = evaluate()
        if env_type in ["humanoid_bench", "isaaclab", "mtbench"]:
            obs = envs.reset()
        if args.use_wandb:
            wandb.log(
                {
                    "eval_avg_return": eval_avg_return,
                    "eval_avg_length": eval_avg_length,
                    "frame": global_step * args.num_envs,
                },
                step=global_step,
            )

    save_params(
        global_step,
        actor,
        qnet,
        qnet_target,
        obs_normalizer,
        critic_obs_normalizer,
        args,
        f"models/{run_name}_final.pt",
    )


if __name__ == "__main__":
    main()
