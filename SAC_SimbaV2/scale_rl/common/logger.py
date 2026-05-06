import os
import re
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

import wandb


def _build_run_name(cfg, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    if not metadata:
        return None

    try:
        total = metadata["total_params"] / 1e6
        actor_total = metadata["actor_params"] / 1e6
        critic_total = metadata["critic_params"] / 1e6
        actor_trainable = metadata.get("actor_trainable_params", 0) / 1e6
        critic_trainable = metadata.get("critic_trainable_params", 0) / 1e6
    except (KeyError, TypeError):
        return None

    algo = metadata.get("algo", "algo")
    env_name = metadata.get("env_name")
    if env_name is None:
        try:
            env_name = cfg.env.env_name
        except AttributeError:
            env_name = "env"
    seed = metadata.get("seed", getattr(cfg, "seed", 0))
    actor_blocks = metadata.get("actor_blocks", "")
    actor_width = metadata.get("actor_width", "")
    critic_blocks = metadata.get("critic_blocks", "")
    critic_width = metadata.get("critic_width", "")
    actor_lora = metadata.get("actor_low_rank", "none")
    critic_lora = metadata.get("critic_low_rank", "none")

    def parse_lora_str(lora_str):
        if lora_str == "none":
            return ""
        parts = lora_str.split("|")
        valid_parts = []
        for part in parts:
            if ":none" not in part:
                valid_parts.append(part)
        if not valid_parts:
            return ""
        return "|".join(valid_parts)

    actor_lora_str = parse_lora_str(actor_lora)
    critic_lora_str = parse_lora_str(critic_lora)

    # Get LoRA params
    lora_alpha = None
    actor_wd = None
    critic_wd = None
    lora_lr_init = None
    lora_lr_end = None
    critic_weight_l2 = None
    kernel_init_scale = None
    try:
        lora_alpha = cfg.agent.lora_alpha
        actor_wd = cfg.agent.actor_LoRA_weight_decay
        critic_wd = cfg.agent.critic_LoRA_weight_decay
        lora_lr_init = cfg.agent.lora_learning_rate_init
        lora_lr_end = cfg.agent.lora_learning_rate_end
        critic_weight_l2 = cfg.agent.critic_weight_L2norm
        kernel_init_scale = cfg.agent.kernel_init_scale
    except AttributeError:
        pass

    run_name = f"{env_name}_{seed}"

    # Actor Info
    if actor_lora_str:
        run_name += f"_AP{actor_total:.2f}M_Tr{actor_trainable:.2f}M"
        if lora_alpha is not None:
            actor_lora_str += f"|alpha:{lora_alpha}"
        if actor_wd is not None:
            actor_lora_str += f"|wd:{actor_wd}"
        if lora_lr_init is not None:
            actor_lora_str += f"|lr_i:{lora_lr_init}"
        if lora_lr_end is not None:
            actor_lora_str += f"|lr_e:{lora_lr_end}"
        run_name += f"_ActLowR={actor_lora_str}"
    else:
        run_name += f"_AP{actor_total:.2f}M"

    # Critic Info
    if critic_lora_str:
        run_name += f"_CP{critic_total:.2f}M_Tr{critic_trainable:.2f}M"
        if lora_alpha is not None:
            critic_lora_str += f"|alpha:{lora_alpha}"
        if critic_wd is not None:
            critic_lora_str += f"|wd:{critic_wd}"
        if lora_lr_init is not None:
            critic_lora_str += f"|lr_i:{lora_lr_init}"
        if lora_lr_end is not None:
            critic_lora_str += f"|lr_e:{lora_lr_end}"
        if critic_weight_l2 is not None:
            l2_str = "true" if bool(critic_weight_l2) else "false"
            critic_lora_str += f"|L2:{l2_str}"
        run_name += f"_CriLowR={critic_lora_str}"
    else:
        run_name += f"_CP{critic_total:.2f}M"

    # Architecture Info
    run_name += f"_AW{actor_width}_AD{actor_blocks}"
    run_name += f"_CW{critic_width}_CD{critic_blocks}"

    # LoRA LR schedule type
    if actor_lora_str or critic_lora_str:
        try:
            lora_sched = str(cfg.agent.lora_lr_schedule)
        except AttributeError:
            lora_sched = "cosine"
        run_name += f"_Sched:{lora_sched}"

    # Kernel init scale
    if (actor_lora_str or critic_lora_str) and kernel_init_scale is not None:
        run_name += f"_k_scale:{kernel_init_scale}"

    return run_name


class WandbTrainerLogger(object):
    def __init__(
        self, cfg: Dict, run_metadata: Optional[Dict[str, Any]] = None,
        wandb_dir: Optional[str] = None,
    ):
        self.cfg = cfg
        dict_cfg = OmegaConf.to_container(cfg, throw_on_missing=True)

        # Ensure env.env_name is in config for grouping
        if run_metadata and "env_name" in run_metadata:
            if "env" not in dict_cfg:
                dict_cfg["env"] = {}
            dict_cfg["env"]["env_name"] = run_metadata["env_name"]

        run_name = _build_run_name(cfg, run_metadata)

        init_kwargs = dict(
            project=cfg.project_name,
            entity=cfg.entity_name,
            group=cfg.group_name,
            config=dict_cfg,
            name=run_name,
        )
        if wandb_dir is not None:
            os.makedirs(wandb_dir, exist_ok=True)
            init_kwargs["dir"] = wandb_dir

        self.run = wandb.init(**init_kwargs)
        # wandb.run is None in offline mode, so retain explicit handle
        if self.run is None:
            self.run = wandb.run
        if self.run is not None and run_name:
            self._ensure_named_run_dir(run_name)

        self.reset()

    def update_metric(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, float) or isinstance(v, int):
                self.average_meter_dict.update(k, v)
            else:
                self.media_dict[k] = v

    def log_metric(self, step: int) -> Dict:
        log_data = {}
        log_data.update(self.average_meter_dict.averages())
        log_data.update(self.media_dict)
        wandb.log(log_data, step=step)

    def reset(self) -> None:
        self.average_meter_dict = AverageMeterDict()
        self.media_dict = {}

    def finish(self) -> None:
        if self.run:
            self.run.finish()

    def _ensure_named_run_dir(self, run_name: str) -> None:
        run_dir = getattr(self.run, "dir", None)
        if not run_dir:
            return
        parent = os.path.dirname(os.path.abspath(run_dir))
        safe_name = re.sub(r"[^A-Za-z0-9._=-]", "_", run_name)
        named_dir = os.path.join(parent, safe_name)
        if os.path.exists(named_dir):
            return
        try:
            os.symlink(run_dir, named_dir)
        except FileExistsError:
            pass
        except OSError:
            pass


class AverageMeterDict(object):
    """
    Manages a collection of AverageMeter instances,
    allowing for grouped tracking and averaging of multiple metrics.
    """

    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string="{}"):
        return {
            format_string.format(name): meter.val for name, meter in self.meters.items()
        }

    def averages(self, format_string="{}"):
        return {
            format_string.format(name): meter.avg for name, meter in self.meters.items()
        }


class AverageMeter(object):
    """
    Tracks and calculates the average and current values of a series of numbers.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )
