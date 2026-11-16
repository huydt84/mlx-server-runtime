"""One-file native MLX architecture implementations."""

from .gemma3 import EntryClass as Gemma3ForCausalLM
from .lfm2 import EntryClass as Lfm2MoeForCausalLM
from .qwen2 import EntryClass as Qwen2ForCausalLM
from .qwen3 import EntryClass as Qwen3ForCausalLM

__all__ = [
    "Gemma3ForCausalLM",
    "Lfm2MoeForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
]
