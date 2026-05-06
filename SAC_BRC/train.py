import os

os.environ['MUJOCO_GL'] = 'egl'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import tqdm
from absl import app, flags

from jaxrl.agent.brc_learner import BRC
from jaxrl.replay_buffer import ParallelReplayBuffer
from jaxrl.envs import ParallelEnv
from jaxrl.normalizer import RewardNormalizer
from jaxrl.logger import EpisodeRecorder
from jaxrl.env_names import get_environment_list

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 1024, 'Mini batch size.')
flags.DEFINE_integer('max_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', 1000000, 'Replay buffer size.')
flags.DEFINE_integer('start_training', 5000,'Number of training steps to start training.')
flags.DEFINE_string('env_names', 'cheetah-run', 'Environment name.')
flags.DEFINE_boolean('log_to_wandb', True, 'Whether to log to wandb.')
flags.DEFINE_boolean('offline_evaluation', True, 'Whether to perform evaluations with temperature=0.')
flags.DEFINE_boolean('render', True, 'Whether to log the rendering to wandb.')
flags.DEFINE_integer('updates_per_step', 2, 'Number of updates per step.')
flags.DEFINE_integer('width_critic', 4096, 'Width of the critic network.')
flags.DEFINE_integer('width_actor', 256, 'Width of the actor network.')
flags.DEFINE_integer('depth_critic', 2, 'Depth (number of residual blocks) of the critic.')

# Method selection
flags.DEFINE_string('method', 'lora', 'Training method: base or lora.')

# Wandb
flags.DEFINE_string('wandb_project', '', 'Wandb project name.')
flags.DEFINE_string('wandb_entity', '', 'Wandb entity name.')

# LoRA parameters (only used when method=lora)
flags.DEFINE_integer('lora_rank', 128, 'Residual-block LoRA rank.')
flags.DEFINE_float('lora_alpha', 128.0, 'Residual-block LoRA alpha scaling factor.')
flags.DEFINE_integer('lora_input_rank', 0, 'LoRA rank for the BroNet input Dense before residual blocks (0=disabled).')
flags.DEFINE_float('lora_input_alpha', -1.0, 'LoRA alpha for the BroNet input Dense (-1=use lora_alpha).')
flags.DEFINE_integer('lora_output_rank', 0, 'LoRA rank for the BroNet output Dense after residual blocks (0=disabled).')
flags.DEFINE_float('lora_output_alpha', -1.0, 'LoRA alpha for the BroNet output Dense (-1=use lora_alpha).')
flags.DEFINE_string('lora_a_init', 'normal', 'LoRA A matrix initializer.')
flags.DEFINE_string('lora_b_init', 'normal', 'LoRA B matrix initializer.')
flags.DEFINE_float('lora_kernel_init_scale', 0.5, 'Scale base weight column norms and LoRA AB effective norm to sqrt(scale) at init (0=no scaling). 0.5 gives ~1/sqrt(2) for each so effective weight ~unit.')
flags.DEFINE_float('lora_weight_decay', 6e-4, 'Weight decay for LoRA A/B params.')

def _format_params(n):
    """Format parameter count as human-readable string (e.g. 1.23M, 456.78K)."""
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def _build_run_name(flags, param_info):
    """Build a descriptive wandb run name (matches FastTD3 convention)."""
    method = flags.method
    parts = [flags.env_names, str(flags.seed)]

    def _resolved_alpha(alpha):
        return flags.lora_alpha if alpha < 0 else alpha

    parts.append(f"AP={_format_params(param_info['actor_total'])}")
    parts.append(f"CP={_format_params(param_info['critic_total'])}")

    if method == 'lora':
        parts.append(f"Tr_C={_format_params(param_info['critic_trainable'])}")
        if flags.lora_input_rank > 0:
            parts.append(f"InLowR=r{flags.lora_input_rank}")
            parts.append(f"InA={_resolved_alpha(flags.lora_input_alpha)}")
        if flags.lora_rank > 0:
            parts.append(f"BlkLowR=r{flags.lora_rank}")
            parts.append(f"BlkA={flags.lora_alpha}")
        if flags.lora_output_rank > 0:
            parts.append(f"OutLowR=r{flags.lora_output_rank}")
            parts.append(f"OutA={_resolved_alpha(flags.lora_output_alpha)}")
        parts.append(f"lora_wd={flags.lora_weight_decay}")
    parts.append(f"CW{flags.width_critic}_CD{flags.depth_critic}")

    return "_".join(parts)


def main(_):
    env_names = get_environment_list(FLAGS.env_names)
    env = ParallelEnv(env_names, seed=FLAGS.seed)
    if FLAGS.offline_evaluation:
        eval_env = ParallelEnv(env_names, seed=FLAGS.seed+42)
    else:
        eval_env = None

    eval_interval = FLAGS.eval_interval if FLAGS.offline_evaluation else 5000

    # Kwargs setup
    kwargs = {}
    kwargs['updates_per_step'] = FLAGS.updates_per_step
    kwargs['width_critic'] = FLAGS.width_critic
    kwargs['width_actor'] = FLAGS.width_actor
    kwargs['depth_critic'] = FLAGS.depth_critic
    kwargs['method'] = FLAGS.method

    # LoRA kwargs
    kwargs['lora_rank'] = FLAGS.lora_rank
    kwargs['lora_alpha'] = FLAGS.lora_alpha
    kwargs['lora_input_rank'] = FLAGS.lora_input_rank
    kwargs['lora_input_alpha'] = FLAGS.lora_input_alpha
    kwargs['lora_output_rank'] = FLAGS.lora_output_rank
    kwargs['lora_output_alpha'] = FLAGS.lora_output_alpha
    kwargs['lora_a_init'] = FLAGS.lora_a_init
    kwargs['lora_b_init'] = FLAGS.lora_b_init
    kwargs['lora_kernel_init_scale'] = FLAGS.lora_kernel_init_scale
    kwargs['lora_weight_decay'] = FLAGS.lora_weight_decay

    num_tasks = len(env.envs)

    agent = BRC(
        FLAGS.seed,
        env.observation_space.sample()[:1],
        env.action_space.sample()[:1],
        num_tasks=num_tasks,
        **kwargs,
    )

    # --- Wandb init (after agent so we can log param counts in name) ---
    if FLAGS.log_to_wandb:
        import wandb
        run_name = _build_run_name(FLAGS, agent.param_info)
        wandb.init(
            config=FLAGS,
            entity=FLAGS.wandb_entity or None,
            project=FLAGS.wandb_project or None,
            group=f'{FLAGS.env_names}',
            name=run_name,
        )
        wandb.config.update(agent.param_info)

    batch_size = 1024 if agent.multitask else 256

    replay_buffer = ParallelReplayBuffer(env.observation_space, env.action_space.shape[-1], FLAGS.replay_buffer_size, num_tasks=num_tasks)

    reward_normalizer = RewardNormalizer(num_tasks, target_entropy=agent.target_entropy, discount=agent.discount)

    statistics_recorder = EpisodeRecorder(num_tasks)

    observations = env.reset()

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1):
        actions = env.action_space.sample() if i < FLAGS.start_training else agent.sample_actions(observations, temperature=1.0)
        next_observations, rewards, terms, truns, goals = env.step(actions)
        reward_normalizer.update(rewards, terms, truns)
        statistics_recorder.update(rewards, goals, terms, truns)
        masks = env.generate_masks(terms, truns)
        replay_buffer.insert(observations, actions, rewards, masks, next_observations)
        observations = next_observations
        observations, terms, truns = env.reset_where_done(observations, terms, truns)
        if i >= FLAGS.start_training:
            batches = replay_buffer.sample(batch_size, FLAGS.updates_per_step)
            batches = reward_normalizer.normalize(batches, agent.get_temperature())
            train_info = agent.update(batches, FLAGS.updates_per_step, i)
            if i % eval_interval == 0 and i >= FLAGS.start_training:
                # Only render video at the final eval step
                is_final = (i + eval_interval > FLAGS.max_steps) or (i == FLAGS.max_steps)
                render_now = FLAGS.render and is_final
                info_dict = statistics_recorder.log(FLAGS, agent, i, eval_env, render=render_now, train_info=train_info)

    if FLAGS.log_to_wandb:
        wandb.finish()


if __name__ == '__main__':
    app.run(main)
