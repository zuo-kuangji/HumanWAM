"""Make tokenizer loading work with HF_HUB_OFFLINE=1 on transformers 4.57.x.

PreTrainedTokenizerBase._patch_mistral_regex calls huggingface_hub.model_info() for every hub-named tokenizer with a
vocab > 100k (to detect Mistral base models). With HF_HUB_OFFLINE=1 that raises OfflineModeIsEnabled even though all
files are in the cache. GR00T's Cosmos-Reason2-2B / Qwen3-VL tokenizer is not a Mistral tokenizer, so when offline the
check is skipped and the tokenizer is returned unchanged. Online behaviour is untouched.
"""
import os

from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB

if not getattr(_PTB, "_hb_offline_fix", False):
    _orig = _PTB._patch_mistral_regex.__func__

    def _patch_mistral_regex(cls, tokenizer, *args, **kwargs):
        try:
            return _orig(cls, tokenizer, *args, **kwargs)
        except Exception as error:
            if os.environ.get("HF_HUB_OFFLINE") == "1" or type(error).__name__ == "OfflineModeIsEnabled":
                return tokenizer
            raise

    _PTB._patch_mistral_regex = classmethod(_patch_mistral_regex)
    _PTB._hb_offline_fix = True
