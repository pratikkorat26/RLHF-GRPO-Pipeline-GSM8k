from __future__ import annotations

import importlib.util


def _probe_vllm() -> str | None:
    if importlib.util.find_spec("vllm") is None:
        return "vLLM is not installed."
    try:
        import vllm._C  # type: ignore[attr-defined]  # noqa: F401
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


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
        vllm_issue = _probe_vllm()
        if vllm_issue is not None:
            import trl.import_utils as trl_import_utils

            trl_import_utils._vllm_available = False
            issues.append(
                f"Disabled broken vLLM integration: {vllm_issue}"
            )

    return issues


def require_vllm(purpose: str = "runtime") -> None:
    """Raise a clear error when the configured code path requires vLLM."""
    issue = _probe_vllm()
    if issue is None:
        return
    raise RuntimeError(
        f"The configured {purpose} path requires vLLM, but the backend is unavailable. "
        "Disable vLLM for this command or install a working Linux vLLM build. "
        f"Original issue: {issue}"
    )
