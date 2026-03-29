from typing import (
    Optional,
    List,
    Type,
    TypeVar,
    cast,
    Tuple,
    Union,
    Iterable,
    Dict,
    Any
)
from glob import glob
from pathlib import Path

from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from olmo.util import PathOrStr, StrEnum
from olmo.exceptions import OLMoConfigurationError

from copy import deepcopy
from dataclasses import dataclass, field, asdict
import torch
from omegaconf import OmegaConf as om, DictConfig, ListConfig
from omegaconf.errors import OmegaConfBaseException


C = TypeVar("C", bound="BaseConfig")
D = TypeVar("D", bound="DictConfig|ListConfig")

ENV = "dev"

class BaseConfig:
    @classmethod
    def _register_resolvers(cls, validate_paths: bool = True):
        # Expands path globs into a list.
        def path_glob(*paths) -> List[str]:
            out = []
            for path in paths:
                matches = sorted(glob(path))
                if not matches and validate_paths:
                    raise FileNotFoundError(f"{path} does not match any files or dirs")
                out.extend(matches)
            return out

        # Chooses the first path in the arguments that exists.
        def path_choose(*paths) -> str:
            from .util import is_url

            for path in paths:
                if is_url(path) or Path(path).exists():
                    return path
            if validate_paths:
                raise FileNotFoundError(", ".join(paths))
            else:
                return ""

        # Finds the latest checkpoint in a folder.
        def path_last_checkpoint(path) -> str:
            from .util import find_latest_checkpoint

            latest_checkpoint = find_latest_checkpoint(path)
            if latest_checkpoint is None:
                if validate_paths:
                    raise FileNotFoundError(f"Could not find a latest checkpoint at {path}")
                else:
                    return ""
            else:
                return str(latest_checkpoint)

        om.register_new_resolver("path.glob", path_glob, replace=True)
        om.register_new_resolver("path.choose", path_choose, replace=True)
        om.register_new_resolver("path.last_checkpoint", path_last_checkpoint, replace=True)

    @classmethod
    def load(
        cls: Type[C],
        path: PathOrStr,
        overrides: Optional[List[str]] = None,
        key: Optional[str] = None,
        validate_paths: bool = True,
    ) -> C:
        """Load from a YAML file."""
        cls._register_resolvers(validate_paths=validate_paths)
        schema = om.structured(cls)
        try:
            raw = om.load(str(path))
            if key is not None:
                raw = raw[key]  # type: ignore
            raw = cls.update_legacy_settings(raw)
            conf = om.merge(schema, raw)
            if overrides:
                conf = om.merge(conf, om.from_dotlist(overrides))
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(str(e))
    
    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        """
        Update the legacy config settings whose schemas have undergone backwards-incompatible changes.
        """
        return config

    def asdict(self, exclude: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        out = asdict(self)  # type: ignore
        if exclude is not None:
            for name in exclude:
                if name in out:
                    del out[name]
        return out

    def save(self, path: PathOrStr) -> None:
        """Save to a YAML file."""
        om.save(config=self, f=str(path))

    def update_with(self, **kwargs):
        result = deepcopy(self)
        for key, value in kwargs.items():
            setattr(result, key, value)
        return result


@dataclass
class ModelConfig(BaseConfig):
    """
    OLMo (model) configuration.
    """

    d_model: int = 2048
    """
    The hidden size of the model.
    """

    n_heads: int = 16
    """
    The number of self-attention heads.
    """

    clip_qkv: Optional[float] = None
    """
    Clip QKV to this value when set.
    """

    n_layers: int = 24
    """
    The number of layers/blocks.
    """

    mlp_hidden_size: Optional[int] = 5504
    """
    Set the exact hidden size for the MLP.
    """

    block_group_size: int = 1
    """
    The number of blocks to group together into a single parent block.
    This has no affect on the number of parameters in the model and is only used to wrap groups
    of blocks together with a single FSDP wrapper during training.

    Compile is only supported with block_group_size 1.
    """

    rope_full_precision: bool = True
    """
    If ``True``, apply RoPE embeddings at full precision regardless of the input type. Otherwise,
    apply RoPE at the precision of the input.
    """

    rope_theta: int = 10_000
    """
    The theta setting for RoPE.
    """

    layer_norm_eps: float = 1e-05

    max_sequence_length: int = 2048
    """
    The maximum input sequence length supported by the model.
    """

    include_bias: bool = False
    """
    Whether or not to include bias parameters in linear layers.
    In PaLM, they got rid of all bias terms because they found that large
    models tend to have near 0 bias terms anyway.
    """

    bias_for_layer_norm: Optional[bool] = False
    """
    Whether or not to include bias parameters in layer norm.
    This is separate from the include_bias parameter, because of a ROCm crash when biases are disabled in
    layer norm.
    When this is None (the default), it inherits the setting from include_bias.
    """

    vocab_size: int = 50280
    """
    Vocabulary size of the model.
    """

    embedding_size: Optional[int] = 50304
    """
    The number of embeddings, i.e. the number of tokens. If set to ``None`` it will default
    to ``vocab_size``. If ``vocab_size`` is not a multiple of 128, setting this to the
    next multiple of 128 that's greater than ``vocab_size`` can improve throughput
    substantially.
    """

    bos_token_id: int = 1
    """
    The ID of the beginning-of-sentence special token.
    """

    eos_token_id: int = 2
    """
    The ID of the end-of-sentence special token.
    """

    pad_token_id: int = 3
    """
    The ID of the token to use for padding. Defaults to the ID of the EOS token.
    """

    init_device: Optional[str] = None
    """
    The torch device to use when initializing the model parameters, e.g. "cpu", "cuda:0", "meta".
    """

    init_cutoff_factor: Optional[float] = None
    """
    A positive factor used to scale the cutoff values when initializing weights with a "fixed distribution" ``init_fn``, such
    as "normal". Setting this to None means values are not cutoff.
    """

    precision: Optional[str] = None
    """
    Precision used to train/evaluate with. You shouldn't set this directly.
    See :data:`TrainConfig.precision` instead.
    """

    emb_init_std: Optional[float] = None
    """
    Override the standard deviation to use when initializing the embedding weights.
    """


@dataclass
class DataConfig(BaseConfig):
    dataset_path: str = "./"
    memmap_dtype: str = "uint16"
    num_workers: int = 0
    drop_last: bool = True
    pin_memory: bool = True
    prefetch_factor: Optional[int] = 16
    persistent_workers: bool = True
    timeout: int = 0
    seed: Optional[int] = 3141

    @property
    def effective_memmap_dtype(self):
        import numpy as np
        try:
            # getattr will check this is part of numpy module, while np.dtype will check
            # if this is a valid numpy dtype.
            np.dtype(dtype := getattr(np, self.memmap_dtype))
        except (AttributeError, TypeError) as e:
            raise TypeError(f"Value {self.memmap_dtype} is not a valid numpy type") from e
        return dtype

@dataclass
class OptimizerConfig(BaseConfig):
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-5

    selective_updates: bool = False
    """
    If ``True``, optimizer parameter and state updates are skipped when the corresponding gradient is 0.
    """

    decay_norm_and_bias: bool = False
    decay_embeddings: bool = False 
    metrics_log_interval: Optional[int] = None
    """
    The interval with which to collect and log detailed parameter-specific metrics.
    This only applies when logging to W&B, since these metrics won't be logged to the console.
    If not set, defaults to the wandb `log_interval`.
    """

    record_update_metrics: bool = False
    """
    Whether to record detailed metrics about the optimizer's parameter updates, like the norm and max
    of the update with AdamW.
    """

    def __post_init__(self):
        self.betas = tuple(self.betas)  # type: ignore[assignment]

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        new_config = config.copy()
        if om.is_dict(new_config):
            assert isinstance(new_config, DictConfig)

            if hasattr(new_config, "name") and new_config.name == "decoupled_lionw":
                new_config.name = "lionw"
                if hasattr(new_config, "eps"):
                    del new_config.eps

        return new_config

class SchedulerUnits(StrEnum):
    steps = "steps"
    tokens = "tokens"

class SchedulerType(StrEnum):
    cosine_with_warmup = "cosine_with_warmup"
    linear_with_warmup = "linear_with_warmup"

@dataclass
class SchedulerConfig(BaseConfig):
    name: SchedulerType = SchedulerType.cosine_with_warmup
    units: SchedulerUnits = SchedulerUnits.steps
    t_warmup: Union[int, float] = 100
    t_max: Optional[Union[int, float]] = None
    alpha_f: float = 0.1

    grad_clip_warmup_steps: Optional[Union[int, float]] = None
    """
    The warmup period for which the max grad norm (or norm ratio) will be set to its
    warmup value of `max_grad_norm * grad_clip_warmup_factor`.
    """

    grad_clip_warmup_factor: Optional[float] = None
    """
    The ratio of the max allowed gradient norm (or norm ratio) for clipping during the warmup period
    vs after the warmup period.
    """

    warmup_min_lr: Optional[float] = None
    """
    The starting LR during the warmup period. If not set this defaults to 10% of
    the target LR.
    """

@dataclass
class SingleGPUConfig(BaseConfig):
    device: str = "auto"
    """
    Device to run single-device training.
    """
    
    def get_device(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        elif self.device == "cuda" and not torch.cuda.is_available():
            raise OLMoConfigurationError("CUDA not available.")
        else:
            return torch.device(self.device)

class FSDPWrapStrategy(StrEnum):
    by_block = "by_block"
    """
    Wrap each OLMo block with its own FSDP instance.
    """

    by_block_and_size = "by_block_and_size"
    """
    Like 'by_block' but `wte` and `ff_out` will be wrapped separately as well.
    """

    by_block_group = "by_block_group"
    """
    Wrap each block group together into its own FSDP instance.
    This requires :attr:`~ModelConfig.block_group_size` to be bigger than 1.
    """

    by_block_group_and_size = "by_block_group_and_size"
    """
    Like 'by_block_group' but `wte` and `ff_out` will be wrapped separately as well.
    """

    size_based = "size_based"
    """
    Used PyTorch's default size-based auto wrap policy.
    """

    one_in_two = "one_in_two"
    one_in_three = "one_in_three"
    one_in_four = "one_in_four"
    one_in_five = "one_in_five"

class FSDPPrecision(StrEnum):
    pure = "pure"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, ``reduce_dtype``,
    and ``buffer_dtype`` all set to the autocast precision data type.
    """

    mixed = "mixed"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, and ``buffer_dtype``
    set to the autocast precision data type, while ``reduce_dtype`` is set to fp32.
    """

@dataclass
class FSDPConfig(BaseConfig):
    use_orig_params: bool = True
    """
    This must be ``True`` if using ``compile`` or you want to track the parameter norm during training.
    """

    sharding_strategy: ShardingStrategy = ShardingStrategy.HYBRID_SHARD

    wrapping_strategy: Optional[FSDPWrapStrategy] = FSDPWrapStrategy.by_block
    """
    The wrapping strategy to use. If ``None``, the default, the model is wrapped with a single top-level
    FSDP instance.
    """

    precision: Optional[FSDPPrecision] = FSDPPrecision.mixed

    hybrid_sharding_num_model_replicas: Optional[int] = None
    """
    The number of model instances, when using a hybrid sharding strategy.
    If not ``None``, this must divide the total number of nodes. If ``None``, the default,
    a model instance is used per node (as determined by ``get_world_size() // get_local_world_size()``).
    PyTorch's default HSDP behavior matches this default behavior.
    """

@dataclass
class SpeedMonitorConfig(BaseConfig):
    window_size: int = 20
    gpu_flops_available: Optional[Union[float, int]] = None

class DistributedStrategy(StrEnum):
    ddp = "ddp"
    """
    Wrap OLMo in torch.nn.parallel.DistributedDataParallel to train across ranks.
    """

    fsdp = "fsdp"
    """
    Wrap OLMo in torch.distributed.fsdp.FullyShardedDataParallel to train across ranks.
    """

    single = "single"
    """
    Train on a single device, i.e., do not distribute training. For development and debugging.
    """

class DDPGradSyncMode(StrEnum):
    batch = "batch"
    """
    Synchronize gradients after computation at each bucket only at the last micro-batch.
    This is slightly faster than gradient syncs across each micro-batch but will consume more memory.
    Can use this mode only when `find_unused_params` is set to False.
    """

    micro_batch = "micro_batch"
    """
    Synchronize gradients after computation at each bucket per micro-batch.
    This will be slightly slower than gradient sync at the last micro-batch, but will consume less memory.
    Can use this mode with both option of `find_unused_params` but specifically recommended to use with `find_unused_params`
    set to True, to prevent errors.
    """

@dataclass
class DDPConfig(BaseConfig):
    grad_sync_mode: DDPGradSyncMode = DDPGradSyncMode.batch
    """
    Gradient sync mode for DDP

    Note: When `find_unused_params` is set, set `grad_sync_mode` to `micro_batch` as different micro-batches might activate
    different parts of the model, ex- MOEs.
    """

    find_unused_params: bool = False
    """
    (from torch documentation)

    This mode allows running backward on a subgraph of the model, and DDP finds out which parameters
    are involved in the backward pass by traversing the autograd graph from the model output and marking
    all unused parameters as ready for reduction. Note that traversing the autograd graph introduces extra overheads,
    so applications should only set find_unused_parameters to True when necessary.
    """

class ShardedCheckpointerType(StrEnum):
    torch_new = "torch_new"
    torch_legacy = "torch_legacy"
    local = "local"

class CheckpointType(StrEnum):
    sharded = "sharded"
    unsharded = "unsharded"
    sharded_ephemeral = "sharded_ephemeral"

@dataclass
class CompilerConfig(BaseConfig):
    mode: Optional[str] = "default"
    """
    The mode to compile the model in. At the moment this can be "default",
    "reduce-overhead" (useful for smaller models/batches), or "max-autotune"
    (the fastest for larger models, but takes a long time to compile).
    """

    fullgraph: bool = False
    """
    Whether it is OK to break model into several subgraphs when compiling.
    Note that this is not compatible with FSDP.
    """

    backend: str = "inductor"
    """
    The backend to use.
    """

    dynamic: Optional[bool] = None
    """
    From the torch docs:
    
    Use dynamic shape tracing. When this is True, we will up-front attempt to generate a kernel that is as dynamic
    as possible to avoid recompilations when sizes change. This may not always work as some
    operations/optimizations will force specialization; use TORCH_LOGS=dynamic to debug overspecialization. When
    this is False, we will NEVER generate dynamic kernels, we will always specialize. By default (None), we
    automatically detect if dynamism has occurred and compile a more dynamic kernel upon recompile.
    """

@dataclass
class WandbConfig(BaseConfig):
    project: Optional[str] = None
    entity: Optional[str] = None
    group: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=lambda: ["watching"])
    log_artifacts: bool = False
    rank_zero_only: bool = True
    log_interval: int = 1

class EvaluatorType(StrEnum):
    downstream = "downstream"
    lm = "lm"

@dataclass
class EvaluatorConfig(BaseConfig):
    label: str
    type: EvaluatorType = EvaluatorType.lm
    data: DataConfig = field(default_factory=DataConfig)
    device_eval_batch_size: Optional[int] = None
    subset_num_batches: Optional[int] = None

@dataclass
class TokenizerConfig(BaseConfig):
    filepath: str = "./"

@dataclass
class TrainConfig(BaseConfig):
    """
    OLMo training configuration.
    """

    run_name: Optional[str] = None
    """
    The name of the run.
    """

    seed: int = 3141
    """
    Used to seed all initial RNG states.
    """

    epoch: Optional[int] = None
    """
    Increment this when starting a new epoch.
    """

    dry_run: bool = False
    """
    If ``True``, don't actually train.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    """
    OLMo Model configuration.
    """

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """
    Optimizer configuration.
    """

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """
    Learning rate scheduler configuration.
    """

    data: DataConfig = field(default_factory=DataConfig)
    """
    Training data configuration.
    """

    restore_dataloader: bool = True
    """
    When restarting, restore the data loader to where it left off.
    If you restarting in order to train on a different dataset, set this to ``False``.
    """

    reset_dataloader_epoch_progress_on_load: bool = False
    """
    This flag is active only if `restore_dataloader` is True and a checkpoint is being loaded.
    If set to True, it will reset `global_train_examples_seen_this_epoch` to 0.
    This means the dataloader will start reading the dataset for the current epoch
    from its beginning. However, `global_step` and `global_train_tokens_seen`
    (important for the learning rate scheduler) will still be restored from the checkpoint.
    Useful when continuing training on a new dataset segment (e.g., Part 2 after Part 1)
    while maintaining a continuous learning rate schedule.
    """

    fast_forward_batches: Optional[int] = None
    """
    When restarting, use this to fast-forward the dataloader beyond the last checkpoint.
    This can be useful when restarting due to a loss spike in order to skip the data that
    corresponded to the spike.
    """

    evaluators: List[EvaluatorConfig] = field(default_factory=list)
    """
    Evaluation configurations.
    """

    eval_interval: int = 100
    """
    How often (in terms of batches) to run evaluations.
    """

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    """
    Tokenizer configuration.
    """

    save_folder: str = "./"
    """
    The directory to save checkpoints to.
    """

    remote_save_folder: Optional[str] = None
    """
    A folder in a cloud bucket to upload saved checkpoints to.
    """

    canceled_check_interval: int = 1000
    """
    How often (in batches) to check if the run has been canceled or reached its time limit.
    """

    save_interval: Optional[int] = None
    """
    How often (in terms of steps) to save sharded training state checkpoints.
    """

    save_interval_unsharded: Optional[int] = None
    """
    How often (if at all) to save unsharded training state checkpoint.
    For large models it can be costly to save these, so it usually makes sense to save
    these less often than regular (sharded) training checkpoints.
    """

    save_steps_unsharded: Optional[List[int]] = None
    """
    A list of steps to save unsharded training state checkpoints.
    This is useful when you want to save unsharded checkpoints at specific steps.
    """

    save_interval_ephemeral: Optional[int] = None
    """
    How often (if at all) to save ephemeral sharded checkpoints. These checkpoints are the same
    as those saved every `save_interval` except that at most only the most recent one of these is kept.
    This is useful when you want to checkpoint often for restarts in case of failures, but don't
    want to keep the majority of these checkpoints.

    For example, suppose you want to keep your checkpoints at every 1000 steps, but you also want to save
    a temporary checkpoint every 100 steps in case your job fails. In that case you would
    set `save_interval=1000` and `save_interval_ephemeral=100`.
    """

    save_num_checkpoints_to_keep: int = -1
    """
    How many sharded checkpoints to keep.
    """

    save_num_unsharded_checkpoints_to_keep: int = -1
    """
    How many unsharded checkpoints to keep.
    """

    save_overwrite: bool = False
    """
    If ``True``, overwrite any conflicting checkpoint files.
    """

    force_save_unsharded: bool = False
    """
    Save an unsharded checkpoint before training (even during a dry run).
    Use this option with `--load-path={PATH}` and `--dry_run` to convert a sharded
    checkpoint into an unsharded checkpoint.
    """

    no_pre_train_checkpoint: bool = False
    """
    Skip saving pre-train checkpoint.
    """

    load_path: Optional[str] = None
    """
    The path to a training checkpoint to restore/resume from. If not set, then training begins from scratch.

    Note that you can make use of the "path.last_checkpoint" Omegaconfig YAML resolver here, which takes
    a local or remote directory and resolves to the latest checkpoint (sharded or unsharded) in that directory.
    For example,

    ```bash
    --load_path='${path.last_checkpoint:s3://ai2-llm/checkpoints/7b/v1_5-mix-run-001}'
    ```

    If `try_load_latest_save` is set and saved checkpoints exist, then `load_path` will be overriden
    by the latest saved checkpoint.
    """

    load_path_sharded_checkpointer: Optional[ShardedCheckpointerType] = None
    """
    The sharded checkpointer type to use to load the initial checkpoint from ``load_path``.
    """

    try_load_latest_save: bool = False
    """
    If set, then training will be resumed from the latest checkpoint in the local save folder, falling
    back to the latest checkpoint in the remote save folder if none exists. If there are no checkpoints
    in the local and remote save folders, then checkpoint loading will fall back to `load_path`.
    """

    reset_optimizer_state: bool = False
    """
    When this is set, we restore the model from a checkpoint (if given), but we leave the optimizer uninitialized.
    We also set a new learning rate schedule that does a new warmup, such that it intercepts the original learning
    curve (according to the current learning rate schedule settings), and continues from there.
    """

    reset_trainer_state: bool = False
    """
    When this is set we don't restore the trainer state from a checkpoint.
    """

    sharded_checkpointer: ShardedCheckpointerType = ShardedCheckpointerType.local
    """
    The name of the sharded checkpointer to use to save (sharded) checkpoints throughout training.
    """

    max_duration: Union[int, str] = 10000
    """
    How long to train for. Used if stop_at is None with 10 extra steps happening after.

    If specified without a unit (the default), the units are assumed to be steps.
    You can also specify this in terms of tokens, for example: `max_duration="2e12T"` means train until
    2 trillion tokens.
    """

    global_train_batch_size: int = 512
    """
    The effective global batch size.
    """

    device_train_batch_size: Optional[int] = None  # calculated automatically
    """
    Don't set this manually. This will be set to ``global_train_batch_size // world_size``.
    """

    device_train_microbatch_size: int = 8
    """
    The number of instances passed to the model in a single forward-backward pass. You should set
    this as large as you can based on available GPU memory.
    """

    device_eval_batch_size: int = 8
    """
    The number of evaluation instances passed to the model in a single forward pass on each device.
    """

    eval_subset_num_batches: int = 1
    """
    The number of batches to use for downstream evaluation from each dataset.
    """

    eval_on_load: bool = False
    """
    When resuming from a checkpoint, run the evaluation loop right away.
    """

    device_train_grad_accum: Optional[int] = None  # calculated automatically
    """
    Don't set this manually. This will be set to ``device_train_batch_size // device_train_microbatch_size``.
    """

    max_grad_norm: Optional[float] = 1.0
    """
    Clip gradient norms to this value if set.
    """

    max_grad_norm_ratio: Optional[float] = None
    """
    If set, gradient norms will be clipped to `max_grad_norm_ratio * exp_avg(norm(grad))`.
    This takes priority over `max_grad_norm` when set.
    """

    precision: Optional[str] = "amp_bf16"
    """
    Precision to train with (e.g. "amp_bf16", "amp_fp16", or "fp32").
    """

    wandb: Optional[WandbConfig] = None
    """
    Weights & Biases configuration.
    """

    speed_monitor: SpeedMonitorConfig = field(default_factory=SpeedMonitorConfig)
    """
    Speed monitor configuration.
    """

    console_log_interval: int = 1
    """
    How often to log to the console.
    """

    gen1_gc_interval: Optional[int] = 1
    """
    How often (in steps) to run generation 1 garbage collection.
    Set to ``None`` to use automatic garbage collection (i.e. we don't mess with it).
    """

    compile: Optional[CompilerConfig] = None
    """
    Settings for compiling the model with ``torch.compile()``.
    """

    distributed_strategy: Optional[DistributedStrategy] = DistributedStrategy.single
    """
    Distributed strategy for OLMo model (eg. single GPU, DDP, FSDP).
    """

    fsdp: Optional[FSDPConfig] = field(default_factory=FSDPConfig)
    """
    Fully sharded data parallel settings.
    """

    ddp: Optional[DDPConfig] = None
    """
    DDP settings.
    """

    single: SingleGPUConfig = field(default_factory=lambda: SingleGPUConfig(device="auto"))
    """
    Single device settings for GPU/CPU/MPS. Defaults to auto-detect the best device.
    """

    softmax_auxiliary_loss: bool = False
    """
    If ``True``, we add the auxiliary loss function from PaLM that encourages the softmax
    normalizing term to be close to 0.
    """

    auxiliary_loss_multiplier: Optional[float] = 1e-4
    """
    Used with `softmax_auxiliary_loss`. PaLM uses 1e-4, Chameleon uses 1e-5.
    """

    time_limit: Optional[float] = None
    """
    The maximum amount of time to train for before saving a checkpoint and ending early.

    Should be in seconds.
    """

    extra_steps_after_cancel: int = 10
    """
    Under certain conditions when a run is canceled we train for a few extra steps after saving
    the final checkpoint so that when the run is restarted from the latest checkpoint we have some
    overlap in metrics.
    """

    early_stopping_factor: Optional[float] = None

    save_data_indices: bool = False
    """
    Save training data indices from each batch for each worker.
    """

    python_profiling: bool = False
    """
    Whether to run the Python profiler on batches 6, 7, and 8.
    """

    torch_profiling: bool = True
    """
    Whether to run the PyTorch profiler on batches 6, 7, and 8.
    """

    stop_at: Optional[int] = None
    """
    Stop at a specific step. Similar to max_duration.
    """

    stop_after: Optional[int] = None
    """
    Stop after a specific number of steps.
    """

    fused_loss: Optional[bool] = False
    """
    Whether to use the fused CE loss function from `flash-attn`.
    """

    module_outputs_save_steps: Optional[List[int]] = None
    """
    Outputs of model submodules are saved during the provided steps. Submodule outputs
    can be compared using `scripts/compare_module_outputs.py`.
    """

    lumi_env: Optional[bool] = False
    """
    The training happens on LUMI.
    """

    @property
    def autocast_precision(self) -> torch.dtype:
        if self.precision == "amp_bf16":
            return torch.bfloat16
        elif self.precision == "amp_fp16":
            return torch.float16
        elif self.precision == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unexpected precision type '{self.precision}'")

    @property
    def fsdp_precision(self) -> Optional[MixedPrecision]:
        if self.fsdp is not None:
            if self.fsdp.precision is None:
                return None
            elif self.fsdp.precision == FSDPPrecision.pure:
                return MixedPrecision(
                    param_dtype=self.autocast_precision,
                    reduce_dtype=self.autocast_precision,
                    buffer_dtype=self.autocast_precision,
                )
            elif self.fsdp.precision == FSDPPrecision.mixed:
                return MixedPrecision(
                    param_dtype=self.autocast_precision,
                    reduce_dtype=torch.float32,
                    buffer_dtype=self.autocast_precision,
                )
            else:
                raise NotImplementedError(f"{self.fsdp.precision}")
        else:
            raise ValueError("self.fsdp is None!")

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        new_config = config.copy()
        if om.is_dict(new_config):
            assert isinstance(new_config, DictConfig)

            if hasattr(new_config, "activation_checkpointing"):
                if new_config.activation_checkpointing is False:
                    new_config.activation_checkpointing = None
                if new_config.activation_checkpointing is True:
                    raise NotImplementedError("Activation checkpointing not implemented")

            if hasattr(new_config, "optimizer"):
                new_config.optimizer = OptimizerConfig.update_legacy_settings(new_config.optimizer)

        return new_config
