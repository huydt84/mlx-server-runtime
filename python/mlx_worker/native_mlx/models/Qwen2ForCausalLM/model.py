"""Qwen2 native MLX model and executor for first forward-pass parity."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn

from ...interfaces import (
    ExecutionBatch,
    NativeBackendOptions,
    StepRequestResult,
    StepResult,
)
from ...mapping import WeightIndex, WeightMappingPlan
from .cache import Qwen2LayerCache, Qwen2RequestCache
from .config import Qwen2ModelConfig
from .weights import load_mapped_weights


def _silu_gate(gate: mx.array, up: mx.array) -> mx.array:
    return nn.silu(gate) * up


class Qwen2Attention(nn.Module):
    """Qwen2 self-attention using MLX built-in SDPA."""

    def __init__(self, config: Qwen2ModelConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
        )

    def __call__(self, x: mx.array, cache: Qwen2LayerCache | None = None) -> mx.array:
        batch_size, seq_len, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(batch_size, seq_len, self.n_heads, -1).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(batch_size, seq_len, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(batch_size, seq_len, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask="causal"
            if seq_len > 1 or (cache is not None and cache.offset > 0)
            else None,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)


class Qwen2Mlp(nn.Module):
    """Qwen2 feed-forward block."""

    def __init__(self, config: Qwen2ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_silu_gate(self.gate_proj(x), self.up_proj(x)))


class Qwen2TransformerBlock(nn.Module):
    """Qwen2 transformer block."""

    def __init__(self, config: Qwen2ModelConfig):
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2Mlp(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x: mx.array, cache: Qwen2LayerCache | None = None) -> mx.array:
        attn_out = self.self_attn(self.input_layernorm(x), cache=cache)
        hidden = x + attn_out
        mlp_out = self.mlp(self.post_attention_layernorm(hidden))
        return hidden + mlp_out


class Qwen2Backbone(nn.Module):
    """Qwen2 decoder backbone."""

    def __init__(self, config: Qwen2ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen2TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self, inputs: mx.array, cache: list[Qwen2LayerCache] | None = None
    ) -> mx.array:
        hidden = self.embed_tokens(inputs)
        active_cache = cache or [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, active_cache):
            hidden = layer(hidden, cache=layer_cache)
        return self.norm(hidden)


class Qwen2ForCausalLm(nn.Module):
    """Qwen2 decoder-only causal LM."""

    def __init__(self, config: Qwen2ModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen2Backbone(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self, inputs: mx.array, cache: list[Qwen2LayerCache] | None = None
    ) -> mx.array:
        hidden = self.model(inputs, cache=cache)
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)


@dataclass
class Qwen2NativeMlxExecutor:
    """Native MLX executor for Qwen2 first-forward-pass parity."""

    model_path: Path
    model_config: Qwen2ModelConfig
    weight_plan: WeightMappingPlan
    weight_index: WeightIndex
    model: Qwen2ForCausalLm = field(init=False)
    _request_caches: dict[str, Qwen2RequestCache] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self.model = Qwen2ForCausalLm(self.model_config)
        mapped_weights = load_mapped_weights(self.weight_index, self.weight_plan)
        self._apply_quantization(mapped_weights)
        self.model.load_weights(mapped_weights, strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())

    def load(self, options: NativeBackendOptions) -> None:
        if options.architecture_class != self.model_config.architecture_class:
            raise ValueError(
                "executor options architecture does not match parsed Qwen2 config"
            )

    def create_cache(self, request_id: str) -> str:
        handle = f"qwen2-cache-{request_id}-{uuid.uuid4().hex}"
        self._request_caches[handle] = Qwen2RequestCache(
            request_id=request_id,
            layers=[
                Qwen2LayerCache() for _ in range(self.model_config.num_hidden_layers)
            ],
        )
        return handle

    def prefill_batch(self, batch: ExecutionBatch) -> StepResult:
        started = time.perf_counter()
        results: list[StepRequestResult] = []
        try:
            for request in batch.requests:
                cache = self._validated_cache(request)
                expected_offset = cache.size()
                if len(request.token_ids) == 0:
                    raise ValueError("prefill requires at least one token per request")
                self._validate_positions(request, expected_offset=expected_offset)
                inputs = mx.array([list(request.token_ids)], dtype=mx.int32)
                logits = self.model(inputs, cache=cache.layers)
                mx.eval(logits)
                self._validate_cache(
                    cache,
                    expected_length=expected_offset + len(request.token_ids),
                )
                request_logits = logits[0]
                next_token = int(mx.argmax(request_logits[-1], axis=-1).item())
                results.append(
                    StepRequestResult(
                        request_id=request.request_id,
                        token_ids=request.token_ids,
                        logits=request_logits,
                        cache_handle=request.cache_handle,
                        cache_length=cache.size(),
                        finished=False,
                        next_token_id=next_token,
                    )
                )
        except Exception:
            for request in batch.requests:
                self.release(request.cache_handle)
            raise

        return StepResult(
            phase=batch.phase,
            results=tuple(results),
            step_time_ms=max(1, int((time.perf_counter() - started) * 1000)),
        )

    def decode_batch(self, batch: ExecutionBatch) -> StepResult:
        started = time.perf_counter()
        results: list[StepRequestResult] = []
        try:
            for request in batch.requests:
                if len(request.token_ids) != 1:
                    raise ValueError("decode requires exactly one token per request")
                cache = self._validated_cache(request)
                expected_offset = cache.size()
                if expected_offset == 0:
                    raise ValueError("decode requires existing prefill state")
                self._validate_positions(request, expected_offset=expected_offset)
                inputs = mx.array([list(request.token_ids)], dtype=mx.int32)
                logits = self.model(inputs, cache=cache.layers)
                mx.eval(logits)
                self._validate_cache(cache, expected_length=expected_offset + 1)
                request_logits = logits[0]
                next_token = int(mx.argmax(request_logits[-1], axis=-1).item())
                results.append(
                    StepRequestResult(
                        request_id=request.request_id,
                        token_ids=request.token_ids,
                        logits=request_logits,
                        cache_handle=request.cache_handle,
                        cache_length=cache.size(),
                        finished=False,
                        next_token_id=next_token,
                    )
                )
        except Exception:
            for request in batch.requests:
                self.release(request.cache_handle)
            raise

        return StepResult(
            phase=batch.phase,
            results=tuple(results),
            step_time_ms=max(1, int((time.perf_counter() - started) * 1000)),
        )

    def cache_len(self, cache_handle: str | None) -> int:
        if cache_handle is None or cache_handle not in self._request_caches:
            return 0
        return self._request_caches[cache_handle].size()

    def release(self, cache_handle: str | None) -> None:
        if cache_handle is None:
            return None
        self._request_caches.pop(cache_handle, None)
        return None

    def forward_token_ids(self, token_ids: list[int]) -> mx.array:
        inputs = mx.array([token_ids], dtype=mx.int32)
        logits = self.model(inputs)
        mx.eval(logits)
        return logits

    def prefill_then_decode_tokens(
        self, prompt_token_ids: Sequence[int], decode_steps: int
    ) -> tuple[list[int], list[int], int]:
        request_id = "parity-request"
        handle = self.create_cache(request_id)
        try:
            prefill = self.prefill_batch(
                ExecutionBatch(
                    phase="prefill",
                    requests=(
                        self._request(
                            request_id=request_id,
                            token_ids=tuple(int(token) for token in prompt_token_ids),
                            positions=tuple(range(len(prompt_token_ids))),
                            cache_handle=handle,
                        ),
                    ),
                )
            )
            tokens = [int(prefill.results[0].next_token_id)]
            lengths = [prefill.results[0].cache_length]
            last_token = tokens[-1]
            for _ in range(decode_steps):
                decode = self.decode_batch(
                    ExecutionBatch(
                        phase="decode",
                        requests=(
                            self._request(
                                request_id=request_id,
                                token_ids=(last_token,),
                                positions=(self.cache_len(handle),),
                                cache_handle=handle,
                            ),
                        ),
                    )
                )
                last_token = int(decode.results[0].next_token_id)
                tokens.append(last_token)
                lengths.append(decode.results[0].cache_length)
            return tokens, lengths, prefill.step_time_ms
        finally:
            self.release(handle)

    def _apply_quantization(self, mapped_weights: list[tuple[str, mx.array]]) -> None:
        quantization = self.model_config.quantization
        if quantization is None:
            return

        weight_names = {name for name, _ in mapped_weights}

        def class_predicate(path: str, module: nn.Module) -> bool:
            if not hasattr(module, "to_quantized"):
                return False
            return f"{path}.scales" in weight_names

        nn.quantize(
            self.model,
            group_size=int(quantization["group_size"]),
            bits=int(quantization["bits"]),
            mode=str(quantization.get("mode", "affine")),
            class_predicate=class_predicate,
        )

    def _stack_batch_inputs(self, batch: ExecutionBatch) -> mx.array:
        if not batch.requests:
            raise ValueError("execution batch must contain at least one request")
        token_lengths = {len(request.token_ids) for request in batch.requests}
        if len(token_lengths) != 1:
            raise ValueError(
                "Phase 3 executor requires equal-length token batches for direct forward"
            )
        return mx.array(
            [list(request.token_ids) for request in batch.requests], dtype=mx.int32
        )

    def _validated_cache(self, request) -> Qwen2RequestCache:
        cache_handle = request.cache_handle
        if cache_handle is None or cache_handle not in self._request_caches:
            raise ValueError("invalid cache handle")
        cache = self._request_caches[cache_handle]
        if cache.request_id != request.request_id:
            raise ValueError("cache handle belongs to different request")
        return cache

    def _validate_positions(self, request, expected_offset: int) -> None:
        if len(request.token_ids) != len(request.positions):
            raise ValueError("token_ids and positions length mismatch")
        expected_positions = tuple(
            range(expected_offset, expected_offset + len(request.token_ids))
        )
        if request.positions != expected_positions:
            raise ValueError("positions do not match cache length")

    def _validate_cache(self, cache: Qwen2RequestCache, expected_length: int) -> None:
        if cache.size() != expected_length:
            raise ValueError("cache length mismatch after update")
        for layer in cache.layers:
            if layer.keys is None or layer.values is None:
                raise ValueError("cache layer missing KV state")
            if int(layer.keys.shape[2]) != expected_length:
                raise ValueError("cache key length mismatch")
            if int(layer.values.shape[2]) != expected_length:
                raise ValueError("cache value length mismatch")

    def _request(
        self,
        *,
        request_id: str,
        token_ids: tuple[int, ...],
        positions: tuple[int, ...],
        cache_handle: str,
    ):
        from ...interfaces import ExecutionRequest

        return ExecutionRequest(
            request_id=request_id,
            token_ids=token_ids,
            positions=positions,
            cache_handle=cache_handle,
            max_new_tokens=1,
            temperature=0.0,
            top_p=1.0,
        )
