from typing import Dict, List, Union

import torch
from torchmetrics import MeanMetric, Metric

from olmo.config import EvaluatorConfig, EvaluatorType, TrainConfig
from olmo.exceptions import OLMoConfigurationError
from olmo.eval.evaluator import Evaluator

from olmo.tokenizer import Tokenizer

__all__ = [
    "Evaluator",
    "ICLMetric",
    "label_to_task_map",
    "build_downstream_evaluator",
    "build_evaluator",
    "build_evaluators",
]

def build_evaluator(
    train_config: TrainConfig, eval_config: EvaluatorConfig, tokenizer: Tokenizer, device: torch.device
) -> Evaluator:
    from ..data import build_eval_dataloader

    if eval_config.type == EvaluatorType.downstream:
        raise NotImplementedError("Downstream evaluation is not implemented currently")
    elif eval_config.type == EvaluatorType.lm:
        # Language modeling evaluation.
        eval_loader = build_eval_dataloader(
            train_config,
            eval_config.data,
            eval_config.device_eval_batch_size or train_config.device_eval_batch_size,
        )

        def make_metric():
            return MeanMetric(nan_strategy="error").to(device)

        eval_metric: Union[Metric, Dict[str, Metric]]
        if eval_config.data.dataset_path:
            eval_metric = make_metric()
        else:
            raise OLMoConfigurationError("Dataset path (dataset_path) is required")

        return Evaluator(
            label=eval_config.label,
            type=eval_config.type,
            eval_loader=eval_loader,
            eval_metric=eval_metric,
            subset_num_batches=eval_config.subset_num_batches,
        )
    else:
        raise ValueError(f"Unexpected evaluator type '{eval_config.type}'")


def build_evaluators(cfg: TrainConfig, device: torch.device) -> List[Evaluator]:
    evaluators = []
    for eval_cfg in cfg.evaluators:
        evaluators.append(build_evaluator(cfg, eval_cfg, None, device))
    return evaluators
