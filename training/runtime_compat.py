from __future__ import annotations

import importlib.util


def prepare_trl_runtime() -> list[str]:
    """Disable optional integrations that are installed but unusable."""
    issues: list[str] = []

    import transformers
    from transformers.integrations import integration_utils as tf_integration_utils

    if importlib.util.find_spec("trackio") is not None:
        try:
            import trackio  # noqa: F401
        except Exception as exc:
            tf_integration_utils.is_trackio_available = lambda: False
            transformers.is_trackio_available = lambda: False
            issues.append(
                f"Disabled broken trackio integration: {type(exc).__name__}: {exc}"
            )

    if importlib.util.find_spec("vllm") is not None:
        try:
            import vllm._C  # type: ignore[attr-defined]  # noqa: F401
        except Exception as exc:
            import trl.import_utils as trl_import_utils

            trl_import_utils._vllm_available = False
            issues.append(
                f"Disabled broken vLLM integration: {type(exc).__name__}: {exc}"
            )

    return issues
