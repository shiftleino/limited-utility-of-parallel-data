"""
Adapted from
[OLMo](https://github.com/allenai/OLMo.git)

Originally adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import MutableMapping
from typing import (
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union
)
from os import PathLike

PathOrStr = Union[str, PathLike]

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from olmo.config import (
    ModelConfig,
    FSDPWrapStrategy
)
from olmo.exceptions import OLMoConfigurationError
from olmo.initialization import init_normal
from olmo.torch_util import ensure_finite_
from olmo.config import TrainConfig, CheckpointType

__all__ = [
    "LayerNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "OLMoBlock",
    "OLMo",
    "OLMoOutput",
    "OLMoGenerateOutput",
]

log = logging.getLogger(__name__)


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")


class LayerNorm(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.eps = config.layer_norm_eps
        self.normalized_shape = (size or config.d_model,)

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, weight=None, bias=None, eps=self.eps)


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (
                self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


class OLMoBlock(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = config.mlp_hidden_size
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        # Make sure QKV clip coefficient is positive, otherwise it's not well-defined.
        if config.clip_qkv is not None:
            assert config.clip_qkv > 0

        # Attention input projection. Projects x -> (q, k, v)
        self.fused_dims = (
            config.d_model,
            config.d_model,
            config.d_model,
        )
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias, device=config.init_device
        )

        # Attention output projection.
        self.attn_out = nn.Linear(
            config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, 2*self.hidden_size, bias=config.include_bias, device=config.init_device
        )

        # Activation function.
        self.act = SwiGLU()

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            self.hidden_size,
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        self.rotary_emb = RotaryEmbedding(config, self.__cache)

        # Layer norms.
        self.attn_norm = LayerNorm(config, size=config.d_model)
        self.ff_norm = LayerNorm(config, size=config.d_model)

    def reset_parameters(self):

        # Mitchell initialization
        proj_in_std = 1 / math.sqrt(self.config.d_model)
        attn_out_std = 1 / (math.sqrt(2 * self.config.d_model * (self.layer_id + 1)))
        ff_out_std = 1 / (math.sqrt(2 * self.ff_out.in_features * (self.layer_id + 1)))
        cutoff_factor = self.config.init_cutoff_factor or 3.0

        init_normal(self.att_proj, proj_in_std, cutoff_factor)
        init_normal(self.ff_proj, proj_in_std, cutoff_factor)
        init_normal(self.attn_out, std=attn_out_std, init_cutoff_factor=cutoff_factor)
        init_normal(self.ff_out, std=ff_out_std, init_cutoff_factor=cutoff_factor)

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        elif bias.device.type == "mps":
            target_dtype = torch.get_autocast_dtype("mps")
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None
        key_len, query_len = k.shape[-2], q.shape[-2]
        
        # Apply rotary embeddings.
        q, k = self.rotary_emb(q, k)

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len:key_len, :key_len], dtype
            )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        # With PyTorch >=2.3 the optimal attention implementation is automatically selected (incl. FlashAttention-2).
        att = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            is_causal=attention_bias is None
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape q, k, v: (batch_size, seq_len, d_model)

        # apply norm before
        h = self.attn_norm(x)

        qkv = self.att_proj(h)

        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        q, k, v = qkv.split(self.fused_dims, dim=-1)

        # Get attention scores.
        att, cache = self.attention(
            q,
            k,
            v,
            attention_bias,
            layer_past=layer_past,
            use_cache=use_cache
        )

        # Add attention scores.
        # shape: (B, T, C)
        x = x + att

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        x = self.ff_norm(x)
        x = self.ff_proj(x)
        x = self.act(x)
        x = self.ff_out(x)
        x = og_x + x

        return x, cache


class OLMoOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """

    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    """
    Attention keys and values from each block.
    """

    hidden_states: Optional[Tuple[torch.Tensor, ...]]
    """
    Hidden states from each block.
    """


class OLMoGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    """
    The generated token IDs.
    """


class OLMoBlockGroup(nn.ModuleList):
    def __init__(self, config: ModelConfig, layer_offset: int, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layers_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        for block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[block_idx]
            block_idx += self.layer_offset

            # shape: (batch_size, seq_len, d_model)
            x, cache = block(
                x,
                attention_bias=attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
            )
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        return x, attn_key_values

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()


class OLMo(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise OLMoConfigurationError("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise OLMoConfigurationError("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_norm=LayerNorm(config),
                ln_f=LayerNorm(config),
            )
        )

        blocks = [OLMoBlock(i, config, self.__cache) for i in range(config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                OLMoBlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        self.transformer.update(
            {
                "ff_out": nn.Linear(
                    config.d_model,
                    config.embedding_size or config.vocab_size,
                    bias=config.include_bias,
                    device=config.init_device,
                )
            }
        )

        # When `init_device"="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None
        self.__num_bck_flops: Optional[int] = None

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        return device

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        
        wte_std = self.config.emb_init_std or 1.0 / math.sqrt(self.config.d_model)
        wte_cutoff_factor = self.config.init_cutoff_factor or 3.0
        init_normal(self.transformer.wte, std=wte_std, init_cutoff_factor=wte_cutoff_factor)

        ff_out_std = 1 / math.sqrt(self.config.d_model)
        ff_out_cutoff_factor = self.config.init_cutoff_factor or 3.0
        init_normal(self.transformer.ff_out, ff_out_std, ff_out_cutoff_factor)

        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
    ) -> OLMoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked. 
            
            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        :param output_hidden_states: If ``True``, the hidden states of each block are returned.
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore
        x = self.transformer.emb_norm(x)

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        # Merge attention mask with attention bias.
        attention_bias = None
        if (attention_mask is not None or past_key_values is not None):
            attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
          
            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    all_hidden_states.append(x) # Add embedded input (hidden states refer to input hidden states to the layer)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                # shape: (batch_size, seq_len, d_model)
                x, cache = block(
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache
                )

                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, cache = block_group(
                    x,
                    attention_bias=attention_bias,
                    layers_past=layers_past,
                    use_cache=use_cache
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        logits = self.transformer.ff_out(x)  # type: ignore

        return OLMoOutput(
            logits=logits,
            attn_key_values=attn_key_values,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )

    def num_params(self, include_embedding: bool = True) -> int:
        """
        Get the total number of parameters.
        """
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".wte." not in np[0],
                params,
            )
        return sum(p.numel() for _, p in params)

    @property
    def num_fwd_flops(self):
        if self.__num_fwd_flops:
            return self.__num_fwd_flops

        # embedding table is just a lookup in the forward pass
        n_params = self.num_params(include_embedding=False)
        # the number of parameters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params
        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_token = (
            self.config.n_layers * 2 * 2 * (self.config.d_model * self.config.max_sequence_length)
        )
        self.__num_fwd_flops = params_flops_per_token + attn_flops_per_token
        return self.__num_fwd_flops

    @property
    def num_bck_flops(self):
        if self.__num_bck_flops:
            return self.__num_bck_flops

        n_params = self.num_params()
        params_flops_per_token = 4 * n_params
        attn_flops_per_token = self.config.n_layers * 8 * (self.config.d_model * self.config.max_sequence_length)
        self.__num_bck_flops = params_flops_per_token + attn_flops_per_token
        return self.__num_bck_flops

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_steps: int = 10,
        greedy: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        stop_sequences: Optional[List[List[int]]] = None,
        ignore_eos: bool = False,
    ) -> OLMoGenerateOutput:
        """
        Generate token IDs using greedy decoding or sampling methods.
        Works only for single-sample generation.

        :param input_ids: A tensor of shape `(seq_len)`.
        :param attention_mask: An optional tensor of shape `(seq_len)`, the same
            as for the forward method.
        :param max_steps: The maximum number of steps to generate.
        :param greedy: If True, performs greedy decoding and ignores temperature,
            top_k, top_p, repetition_penalty, and no_repeat_ngram_size. Defaults to False.
        :param temperature: The value used to module the next token probabilities
            when `greedy=False`. Must be > 0. Values < 1.0 sharpen the distribution,
            values > 1.0 flatten it.
        :param top_k: The number of highest probability vocabulary tokens to keep for
            top-k-filtering when `greedy=False`. If > 0, only the `top_k` most
            probable tokens are considered. `top_k=0` means no top-k filtering.
        :param top_p: If set to a float < 1.0 and > 0.0 when `greedy=False`,
            only the smallest set of most probable tokens whose cumulative probability
            exceeds `top_p` are kept for generation (nucleus sampling).
            `top_p=1.0` means no top-p filtering.
        :param repetition_penalty: The parameter for repetition penalty. 1.0 means no penalty.
            Values > 1.0 penalize new tokens based on their appearance in the text so far.
            Defaults to 1.0.
        :param no_repeat_ngram_size: If > 0, all n-grams of this size can only occur once.
            Defaults to 0 (no n-gram blocking).
        :param stop_sequences: A list of token IDs that, if generated, will stop the generation.
            If `None`, generation continues until `max_steps` is reached.
        :param ignore_eos: If True, the generation will not stop when an end-of-sequence token is generated and instead the next most likely token is selected.
        """

        if not (input_ids.ndim == 1):
            raise ValueError("input_ids must be a 1D tensor for single-sample generation.")
        if not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0.")
        if not (top_k >= 0):
            raise ValueError("top_k must be non-negative.")
        if not greedy and temperature <= 0.0:
            raise ValueError(
                "Temperature must be positive when greedy=False. "
                "To use greedy-like behavior with temperature 0, set greedy=True."
            )
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be a positive value.")
        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be non-negative.")
        
        stop_sequence_tensors = []
        if stop_sequences is not None:
            for seq in stop_sequences:
                stop_sequence_tensors.append(torch.tensor(seq, device=self.device, dtype=torch.long))

        generated_sequence_tensor = input_ids.clone()
        current_input_ids = input_ids.unsqueeze(0)
        current_attention_mask = attention_mask.unsqueeze(0) if attention_mask is not None else None
        past_key_values = None
        last_token_id = current_input_ids[:, -1].unsqueeze(1)

        for step in range(max_steps):
            if step == 0:
                model_inputs = current_input_ids
                attention_mask_for_model_step = current_attention_mask
            else:
                model_inputs = last_token_id
                attention_mask_for_model_step = current_attention_mask

            output = self.forward(
                model_inputs,
                attention_mask=attention_mask_for_model_step,
                past_key_values=past_key_values,
                use_cache=True,
                last_logits_only=True,
            )

            next_token_logits = output.logits[:, -1, :]

            if repetition_penalty != 1.0 and not greedy:
                for token_id_to_penalize in torch.unique(generated_sequence_tensor):
                    logit_value = next_token_logits[0, token_id_to_penalize]
                    if logit_value > 0:
                        next_token_logits[0, token_id_to_penalize] = logit_value / repetition_penalty
                    else:
                        next_token_logits[0, token_id_to_penalize] = logit_value * repetition_penalty
            
            if no_repeat_ngram_size > 0 and not greedy:
                current_seq_len = generated_sequence_tensor.shape[0]
                if current_seq_len >= no_repeat_ngram_size - 1:
                    banned_tokens_for_next_step = []
                    vocab_size = next_token_logits.shape[-1]

                    if no_repeat_ngram_size == 1:
                        seen_tokens = torch.unique(generated_sequence_tensor)
                        for token_val_tensor in seen_tokens:
                            banned_tokens_for_next_step.append(token_val_tensor.item())
                    else:
                        all_past_ngrams = set()
                        if current_seq_len >= no_repeat_ngram_size:
                            for k_idx in range(current_seq_len - no_repeat_ngram_size + 1):
                                past_ngram = tuple(generated_sequence_tensor[k_idx : k_idx + no_repeat_ngram_size].tolist())
                                all_past_ngrams.add(past_ngram)
                        
                        if all_past_ngrams:
                            prefix_tokens_list = generated_sequence_tensor[current_seq_len - (no_repeat_ngram_size - 1):].tolist()

                            for token_candidate_idx in range(vocab_size):
                                potential_ngram_tuple = tuple(prefix_tokens_list + [token_candidate_idx])
                                if potential_ngram_tuple in all_past_ngrams:
                                    banned_tokens_for_next_step.append(token_candidate_idx)
                    
                    if banned_tokens_for_next_step:
                        for token_to_ban in banned_tokens_for_next_step:
                             if 0 <= token_to_ban < vocab_size:
                                next_token_logits[0, token_to_ban] = -float("inf")
            
            if ignore_eos and hasattr(self.config, 'eos_token_id') and self.config.eos_token_id is not None:
                eos_token_id = self.config.eos_token_id
                next_token_logits[0, eos_token_id] = -float("inf")

            if greedy:
                next_token_id_tensor = torch.argmax(next_token_logits, dim=-1)
            else:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    k_actual = min(top_k, next_token_logits.size(-1))
                    top_k_values, _ = torch.topk(next_token_logits, k=k_actual, dim=-1)
                    kth_value = top_k_values[:, -1].unsqueeze(-1)
                    indices_to_remove = next_token_logits < kth_value
                    next_token_logits[indices_to_remove] = -float('inf')

                if 0.0 < top_p < 1.0:
                    sorted_logits_for_p, sorted_indices_for_p = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits_for_p, dim=-1), dim=-1)
                    num_to_keep = (cumulative_probs < top_p).sum(dim=-1) + 1
                    num_to_keep = torch.clamp(num_to_keep, min=1, max=next_token_logits.size(-1))

                    if num_to_keep.item() < next_token_logits.size(-1):
                        indices_to_discard_in_sorted_order = sorted_indices_for_p[:, num_to_keep.item():]
                        next_token_logits.scatter_(-1, indices_to_discard_in_sorted_order, -float('inf'))
                
                if torch.all(next_token_logits == -float('inf')):
                    # Fallback: if all logits are -inf (e.g., due to very restrictive n-gram or top-k/p)
                    # allow sampling from a uniform distribution over non-inf logits if any, or all if all are -inf.
                    # This part might need more sophisticated handling, e.g. prioritizing less penalized tokens.
                    # For simplicity, if all are -inf, we might pick a random token or signal an issue.
                    # Here, we just let softmax handle it, which will be uniform if all are -inf.
                    # If only some are -inf, softmax will correctly assign 0 prob to them.
                    log.warning("All logits became -inf. Sampling might be from a limited or uniform distribution.")
                    
                    # Fallback if all are -inf -> uniform
                    if torch.all(torch.isneginf(next_token_logits)):
                        next_token_logits = torch.zeros_like(next_token_logits)


                probabilities = F.softmax(next_token_logits, dim=-1)
                
                if torch.isnan(probabilities).any():
                    log.warning("NaN probabilities detected. Falling back to uniform distribution.")
                    probabilities = torch.ones_like(probabilities) / probabilities.size(-1)

                next_token_id_tensor = torch.multinomial(probabilities, num_samples=1).squeeze(-1)

            token_id_item = next_token_id_tensor.item()
            generated_sequence_tensor = torch.cat((generated_sequence_tensor, next_token_id_tensor), dim=0)

            if hasattr(self.config, 'eos_token_id') and token_id_item == self.config.eos_token_id:
                break
            
            should_stop = False
            if stop_sequence_tensors:
                current_len = generated_sequence_tensor.shape[0] - input_ids.shape[0]
                for stop_tensor in stop_sequence_tensors:
                    stop_len = len(stop_tensor)
                    if current_len >= stop_len:
                        if torch.equal(generated_sequence_tensor[-stop_len:], stop_tensor):
                            should_stop = True
                            break
            
            if should_stop:
                break

            last_token_id = next_token_id_tensor.unsqueeze(0)
            past_key_values = output.attn_key_values

            if current_attention_mask is not None:
                current_attention_mask = torch.cat(
                    (current_attention_mask, current_attention_mask.new_ones((1, 1), device=current_attention_mask.device)), dim=-1
                )
        
        return OLMoGenerateOutput(
            token_ids=generated_sequence_tensor
        )

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        if wrap_strategy is None:
            return None

        # The 'recurse' mode for the wrap function does not behave like you'd expect.
        # Even if we return False, it may still recurse because PyTorch does what it wants,
        # not what you want. This causes issues when, for example, we want to wrap 'ff_out' (a linear layer)
        # but not other linear layers within a block.
        # So we have to explicitly tell PyTorch which linear layers to wrap, and we also just
        # return True in 'recurse' mode for simplicity.
        size_based_module_to_wrap = {self.transformer.wte, self.transformer.ff_out}

        if wrap_strategy == FSDPWrapStrategy.by_block:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlock)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_and_size:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, (OLMoBlock,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlockGroup)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group_and_size:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group_and_size' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, (OLMoBlockGroup,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.size_based:
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            return size_based_auto_wrap_policy
        elif wrap_strategy in {
            FSDPWrapStrategy.one_in_two,
            FSDPWrapStrategy.one_in_three,
            FSDPWrapStrategy.one_in_four,
            FSDPWrapStrategy.one_in_five,
        }:
            c = {
                FSDPWrapStrategy.one_in_two: 2,
                FSDPWrapStrategy.one_in_three: 3,
                FSDPWrapStrategy.one_in_four: 4,
                FSDPWrapStrategy.one_in_five: 5,
            }[wrap_strategy]

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlock) and module.layer_id % c == 0
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        else:
            raise NotImplementedError(wrap_strategy)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: PathOrStr, device: str = "cpu", checkpoint_type: Optional[CheckpointType] = None
    ) -> OLMo:
        """
        Load a OLMo model from a checkpoint.
        """
        from .util import resource_path

        # Guess checkpoint type.
        if checkpoint_type is None:
            try:
                if resource_path(checkpoint_dir, "model.pt").is_file():
                    checkpoint_type = CheckpointType.unsharded
                else:
                    checkpoint_type = CheckpointType.sharded
            except FileNotFoundError:
                checkpoint_type = CheckpointType.sharded

        # Load config.
        config_path = resource_path(checkpoint_dir, "config.yaml")
        model_config = ModelConfig.load(config_path, key="model", validate_paths=False)

        if checkpoint_type == CheckpointType.unsharded:
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            model = OLMo(model_config)

            # Load state dict directly to target device.
            state_dict_path = resource_path(checkpoint_dir, "model.pt")
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(model._make_state_dict_compatible(state_dict)[0])
            model = model.to(torch.device(device))
        else:
            train_config = TrainConfig.load(config_path)
            # train_config.sharded_checkpointer == ShardedCheckpointerType.torch_new
            from olmo.checkpoint import load_model_state

            # Initialize model on target device. In this case the state dict is loaded in-place
            # so it's not necessary to start on CPU if the target device is a GPU.
            model_config.init_device = device
            model = OLMo(model_config)

            # Load state dict in place.
            load_model_state(checkpoint_dir, model)

        return model.eval()

    def _make_state_dict_compatible(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Set[str]]]:
        """
        Handles some cases where the state dict is valid yet may need to be transformed in order to
        be loaded.

        This modifies the state dict in-place and also returns it, along with a mapping of original key
        names to new key names in cases where the keys were simply renamed. That mapping can be used
        to make a corresponding optimizer state dict compatible as well.
        """
        import re
        from fnmatch import fnmatch

        new_keys_to_og_keys: Dict[str, str] = {}

        # Remove "_fsdp_wrapped_module." prefix from all keys. We don't want this prefix when the model is
        # not wrapped in FSDP. And when the model is wrapped in FSDP, loading this state dict will still work
        # fine without the prefixes. This also simplifies the other steps below.
        for key in list(state_dict.keys()):
            state_dict[(new_key := key.replace("_fsdp_wrapped_module.", ""))] = state_dict.pop(key)
            new_keys_to_og_keys[new_key] = key


        for key in list(state_dict.keys()):
            if fnmatch(key, "transformer.*.norm.weight"):
                tensor = state_dict.pop(key)
                state_dict[(new_key := key.replace("norm.weight", "attn_norm.weight"))] = tensor
                new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                state_dict[(new_key := key.replace("norm.weight", "ff_norm.weight"))] = tensor.clone()
                new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                del new_keys_to_og_keys[key]
            elif fnmatch(key, "transformer.*.norm.bias"):
                tensor = state_dict.pop(key)
                state_dict[(new_key := key.replace("norm.bias", "attn_norm.bias"))] = tensor
                new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                state_dict[(new_key := key.replace("norm.bias", "ff_norm.bias"))] = tensor.clone()
                new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                del new_keys_to_og_keys[key]

        # For loading a state dict that was saved with a different `block_group_size`.
        if "transformer.block_groups.0.0.attn_out.weight" in state_dict.keys():
            state_dict_block_group_size = len(
                [k for k in state_dict.keys() if fnmatch(k, "transformer.block_groups.0.*.attn_out.weight")]
            )
        else:
            state_dict_block_group_size = 1
        if self.config.block_group_size != state_dict_block_group_size:
            log.info(
                f"Regrouping state dict blocks from group size {state_dict_block_group_size} to "
                f"group size {self.config.block_group_size}"
            )
            # For simplicity we're first going to flatten out the block groups in the state dict (if necessary)
            # and then (re-)group them into the right block sizes.
            if state_dict_block_group_size > 1:
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.block_groups\.(\d+)\.(\d+)\..*", key)) is not None:
                        group_idx, group_block_idx = int(m.group(1)), int(m.group(2))
                        block_idx = (group_idx * state_dict_block_group_size) + group_block_idx
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"block_groups.{group_idx}.{group_block_idx}.", f"blocks.{block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

            if self.config.block_group_size > 1:
                # Group the state dict blocks into the right block size.
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.blocks\.(\d+)\..*", key)) is not None:
                        block_idx = int(m.group(1))
                        group_idx, group_block_idx = (
                            block_idx // self.config.block_group_size,
                            block_idx % self.config.block_group_size,
                        )
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"blocks.{block_idx}.", f"block_groups.{group_idx}.{group_block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

        og_keys_to_new: Dict[str, Set[str]] = defaultdict(set)
        for new_key, og_key in new_keys_to_og_keys.items():
            og_keys_to_new[og_key].add(new_key)

        return state_dict, og_keys_to_new
