import argparse
from pathlib import Path

from code.train_pipeline import DEFAULT_CONFIG, train_and_evaluate


# Central place for training defaults.
# Edit these values directly when you want to change the training setup
# without passing extra command-line arguments.
TRAIN_CONFIG = {
    "representation": DEFAULT_CONFIG["representation"],
    "model": DEFAULT_CONFIG["model"],
    "rounds": DEFAULT_CONFIG["rounds"],
    "random_state": DEFAULT_CONFIG["random_state"],
    "filter_monomer_only": DEFAULT_CONFIG["filter_monomer_only"],
    "rf_n_estimators": DEFAULT_CONFIG["rf_n_estimators"],
    "rf_n_jobs": DEFAULT_CONFIG["rf_n_jobs"],
    "xgb_n_estimators": DEFAULT_CONFIG["xgb_n_estimators"],
    "xgb_max_depth": DEFAULT_CONFIG["xgb_max_depth"],
    "xgb_learning_rate": DEFAULT_CONFIG["xgb_learning_rate"],
    "xgb_tree_method": DEFAULT_CONFIG["xgb_tree_method"],
    "lr_max_iter": DEFAULT_CONFIG["lr_max_iter"],
}


def print_section(title):
    print("\n" + "=" * 70)
    print(f"{title}".center(70))
    print("=" * 70)


def print_step(msg):
    print(f"[INFO] {msg}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate SIF/SGF models")
    parser.add_argument(
        "--train-csv",
        type=str,
        default="./sample_train.csv",
        help="Path to training CSV (default: ./sample_train.csv)",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="./sample_test.csv",
        help="Path to test CSV (default: ./sample_test.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=TRAIN_CONFIG["model"],
        choices=["lr", "rf", "xgb"],
        help="Model type to train",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default=TRAIN_CONFIG["representation"],
        choices=[
            "morgan",
            "avalon",
            "molt5",
            "molformer",
            "avalon+molformer",
            "avalon+molt5",
        ],
        help="Feature representation to use",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=TRAIN_CONFIG["rounds"],
        help="Number of training rounds with different random seeds",
    )

    args = parser.parse_args()

    print_section("TRAIN PIPELINE START")
    print_step(f"Train CSV: {Path(args.train_csv).resolve()}")
    print_step(f"Test CSV: {Path(args.test_csv).resolve()}")
    print_step(f"Model: {args.model}")
    print_step(f"Representation: {args.representation}")
    print_step(f"Rounds: {args.rounds}")

    results = train_and_evaluate(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        config={
            **TRAIN_CONFIG,
            "model": args.model,
            "representation": args.representation,
            "rounds": args.rounds,
        },
    )

    print_section("TRAIN RESULTS")
    for task, result in results.items():
        metrics = result["best_metrics"]
        print(f"{task}:")
        print(f"  Train Samples : {result['train_samples']}")
        print(f"  Test Samples  : {result['test_samples']}")
        print(f"  Best Round    : {result['best_round']}")
        print(f"  ACC           : {metrics['ACC']:.4f}")
        print(f"  F1            : {metrics['F1']:.4f}")
        print(f"  Precision     : {metrics['Precision']:.4f}")
        print(f"  Recall        : {metrics['Recall']:.4f}")
        print(f"  AUPRC         : {metrics['AUPRC']:.4f}")
        print(f"  JSON          : {result['json_path']}")
        print(f"  Model         : {result['model_path']}")

    print("\n" + "=" * 70)
    print("TRAIN PIPELINE COMPLETED".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
