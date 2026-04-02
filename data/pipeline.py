"""Compatibility wrapper forwarding to gsm8k_grpo.data.pipeline."""
from gsm8k_grpo.data.pipeline import *  # noqa: F401,F403
from gsm8k_grpo.cli.pipeline import main
if __name__ == "__main__":
    main()
