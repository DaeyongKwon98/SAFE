"""Workspace-local Python startup hooks.

This file is intentionally inert unless DAEYONG_VLLM_TORCH_PRELOAD=1 is set.
It lets vLLM subprocesses inherit the same torch import-order workaround used by
inference_vllm.py.
"""

import os


if os.environ.get("DAEYONG_VLLM_TORCH_PRELOAD") == "1":
    try:
        import torch._logging  # noqa: F401
        import torch._numpy  # noqa: F401
        from torch._guards import detect_fake_mode as _torch_detect_fake_mode  # noqa: F401
        from torch._logging import LazyString as _TorchLazyString  # noqa: F401
        from torch._dynamo import config as _torch_dynamo_config  # noqa: F401
        from torch._subclasses.fake_tensor import (  # noqa: F401
            FakeTensor as _TorchFakeTensor,
            is_fake as _torch_is_fake,
            maybe_get_fake_mode as _torch_maybe_get_fake_mode,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
