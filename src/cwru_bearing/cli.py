"""Small command-line interface; preprocessing works without importing torch."""
import argparse
import json
from .data import load_config, validate_recordings


def main():
    parser = argparse.ArgumentParser(description="CWRU bearing diagnosis: validate data, generate spectrograms, train, and evaluate")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["validate", "generate", "compare", "train"]:
        p = sub.add_parser(name)
        p.add_argument("--profile", required=True, choices=["stft_a", "stft_b", "cross_load", "cross_fault_size"])
        if name in ["validate", "generate", "compare"]:
            p.add_argument("--data-root", required=True)
        if name in ["generate", "train"]:
            p.add_argument("--output", required=True)
        if name in ["compare", "train"]:
            p.add_argument("--images", required=True)
        if name == "train":
            p.add_argument("--num-workers", type=int, help="Explicit runtime override; may change stochastic execution")
            p.add_argument("--execute-training", action="store_true", help="Start model training; may download ImageNet weights")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        from .training import evaluate_checkpoint
        result = evaluate_checkpoint(args.checkpoint, args.images, args.output)
    else:
        config = load_config(args.profile)
        if args.command == "train" and args.num_workers is not None:
            if args.num_workers < 0:
                parser.error("--num-workers must be nonnegative")
            config["num_workers"] = args.num_workers
        if args.command == "validate":
            result = validate_recordings(args.data_root, config)
        elif args.command == "generate":
            from .preprocessing import generate
            validate_recordings(args.data_root, config)
            result = generate(args.data_root, args.output, config)
        elif args.command == "compare":
            from .preprocessing import compare_historical
            result = compare_historical(args.data_root, args.images, config)
        else:
            if not args.execute_training:
                parser.error("Training requires --execute-training; no training has been started")
            from .training import run_training
            result = run_training(args.images, args.output, config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
