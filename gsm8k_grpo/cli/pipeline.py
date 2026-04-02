"""Package CLI entrypoint for building GSM8K GRPO artifacts."""

from gsm8k_grpo.data.pipeline import build_pipeline, parse_args


def main() -> None:
    args = parse_args()
    build_pipeline(
        splits=args.splits,
        output_dir=args.output_dir,
        system_prompt=args.system_prompt,
        add_difficulty=not args.no_difficulty,
        num_workers=args.num_workers,
        save_jsonl_flag=not args.no_jsonl,
        save_hf_flag=not args.no_hf,
        max_prompt_length=args.max_prompt_length,
        max_parse_error_rate=args.max_parse_error_rate,
        max_truncation_risk_rate=args.max_truncation_risk_rate,
        source_dataset_name=args.source_dataset_name,
        source_dataset_config=args.source_dataset_config,
    )


if __name__ == "__main__":
    main()
