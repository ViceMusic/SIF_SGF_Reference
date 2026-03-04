import sys
import argparse
from pathlib import Path
import pandas as pd
from code.inference import predict_from_csv
from code.inference_base import predict_from_csv as predict_from_csv_base


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
        help="Path to the raw CSV file (default: ./raw.csv)"
    )

    args = parser.parse_args()
    input_path = Path(args.input).resolve()

    print_section("PREDICTION PIPELINE START")
    print_step(f"Input file: {input_path}")

    # ======================== 主模型预测 ========================
    print_section("RUNNING MAIN MODEL")

    result = predict_from_csv(args.input)
    if not result:
        print("[WARNING] No results produced.")
        return

    df = pd.DataFrame(result)

    cwd = Path().resolve()
    sif_dir = cwd / 'SIF_result'
    sgf_dir = cwd / 'SGF_result'
    sif_dir.mkdir(parents=True, exist_ok=True)
    sgf_dir.mkdir(parents=True, exist_ok=True)

    sif_df = df[['SMILES', 'SIF']].copy()
    sgf_df = df[['SMILES', 'SGF']].copy()

    sif_csv = sif_dir / 'sif_results.csv'
    sgf_csv = sgf_dir / 'sgf_results.csv'
    sif_html = sif_dir / 'sif_results.html'
    sgf_html = sgf_dir / 'sgf_results.html'

    sif_df.to_csv(sif_csv, index=False)
    sgf_df.to_csv(sgf_csv, index=False)
    sif_df.to_html(sif_html, index=False)
    sgf_df.to_html(sgf_html, index=False)

    print_step("Main model prediction finished.")
    print_step("Results saved successfully.")

    # ======================== Baseline预测 ========================
    print_section("RUNNING BASELINE MODEL (Morgan)")

    result = predict_from_csv_base(args.input)
    if not result:
        print("[WARNING] No baseline results produced.")
        return

    df = pd.DataFrame(result)

    sif_dir_b = cwd / 'SIF_result_baseline'
    sgf_dir_b = cwd / 'SGF_result_baseline'
    sif_dir_b.mkdir(parents=True, exist_ok=True)
    sgf_dir_b.mkdir(parents=True, exist_ok=True)

    sif_df_b = df[['SMILES', 'SIF']].copy()
    sgf_df_b = df[['SMILES', 'SGF']].copy()

    sif_csv_b = sif_dir_b / 'sif_results.csv'
    sgf_csv_b = sgf_dir_b / 'sgf_results.csv'
    sif_html_b = sif_dir_b / 'sif_results.html'
    sgf_html_b = sgf_dir_b / 'sgf_results.html'

    sif_df_b.to_csv(sif_csv_b, index=False)
    sgf_df_b.to_csv(sgf_csv_b, index=False)
    sif_df_b.to_html(sif_html_b, index=False)
    sgf_df_b.to_html(sgf_html_b, index=False)

    print_step("Baseline model prediction finished.")
    print_step("Results saved successfully.")

    # ======================== 汇总输出 ========================
    print_section("OUTPUT FILES")

    print("Main Model:")
    print(f"  SIF CSV : {sif_csv}")
    print(f"  SIF HTML: {sif_html}")
    print(f"  SGF CSV : {sgf_csv}")
    print(f"  SGF HTML: {sgf_html}")

    print("\nBaseline Model (Morgan):")
    print(f"  SIF CSV : {sif_csv_b}")
    print(f"  SIF HTML: {sif_html_b}")
    print(f"  SGF CSV : {sgf_csv_b}")
    print(f"  SGF HTML: {sgf_html_b}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()