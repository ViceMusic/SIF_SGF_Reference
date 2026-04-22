import argparse
from pathlib import Path
import pandas as pd
from code.inference import predict_from_csv


def print_section(title):
    print("\n" + "=" * 70)
    print(f"{title}".center(70))
    print("=" * 70)


def print_step(msg):
    print(f"[INFO] {msg}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline for feature extraction and prediction"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./raw.csv",
        help="Path to the raw CSV file (default: ./raw.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lr",
        choices=["lr", "rf", "xgb"],
        help="Model type to use (default: lr)",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="avalon+molformer",
        choices=[
            "morgan",
            "avalon",
            "molt5",
            "molformer",
            "avalon+molformer",
            "avalon+molt5",
        ],
        help="Feature representation to use for both SIF and SGF",
    )

    args = parser.parse_args()
    input_path = Path(args.input).resolve()

    print_section("PREDICTION PIPELINE START")
    print_step(f"Input file: {input_path}")
    print_step(f"Model: {args.model}")
    print_step(f"Representation: {args.representation}")

    print_section("RUNNING PREDICTION")
    result = predict_from_csv(
        args.input,
        model_name=args.model,
        representation=args.representation,
    )
    if not result:
        print("[WARNING] No results produced.")
        return

    df = pd.DataFrame(result)

    cwd = Path().resolve()
    sif_dir = cwd / "SIF_result"
    sgf_dir = cwd / "SGF_result"
    sif_dir.mkdir(parents=True, exist_ok=True)
    sgf_dir.mkdir(parents=True, exist_ok=True)

    sif_df = df[["SMILES", "SIF"]].copy()
    sgf_df = df[["SMILES", "SGF"]].copy()

    sif_csv = sif_dir / "sif_results.csv"
    sgf_csv = sgf_dir / "sgf_results.csv"
    sif_html = sif_dir / "sif_results.html"
    sgf_html = sgf_dir / "sgf_results.html"

    sif_df.to_csv(sif_csv, index=False)
    sgf_df.to_csv(sgf_csv, index=False)
    sif_df.to_html(sif_html, index=False)
    sgf_df.to_html(sgf_html, index=False)

    print_step("Prediction finished.")
    print_step("Results saved successfully.")

    print_section("OUTPUT FILES")
    print("Prediction Results:")
    print(f"  SIF CSV : {sif_csv}")
    print(f"  SIF HTML: {sif_html}")
    print(f"  SGF CSV : {sgf_csv}")
    print(f"  SGF HTML: {sgf_html}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
