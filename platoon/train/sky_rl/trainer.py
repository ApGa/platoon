from skyrl_train.entrypoints.main_base import BasePPOExp
from platoon.train.sky_rl.generator import PlatoonSkyRLGenerator
from omegaconf import DictConfig
import ray
from typing import Callable
from platoon.envs.base import Task

class PlatoonSkyRLTrainer(BasePPOExp):
    def __init__(self, cfg: DictConfig, rollout_fn: Callable[[Task, dict], dict], get_task_fn: Callable[[str], Task]):
        super().__init__(cfg)
        self.rollout_fn = rollout_fn
        self.get_task_fn = get_task_fn

    def get_generator(self, cfg, tokenizer, inference_engine_client) -> PlatoonSkyRLGenerator:
        return PlatoonSkyRLGenerator(
            generator_cfg=cfg.generator,
            inference_engine_client=inference_engine_client,
            model_name=cfg.trainer.policy.model.path,
            rollout_fn=self.rollout_fn,
            get_task_fn=self.get_task_fn,
        )
 
@ray.remote(num_cpus=1)
def entrypoint(cfg: DictConfig, rollout_fn: Callable[[Task, dict], dict], get_task_fn: Callable[[str], Task]):
    exp = PlatoonSkyRLTrainer(cfg, rollout_fn, get_task_fn)
    exp.run()
