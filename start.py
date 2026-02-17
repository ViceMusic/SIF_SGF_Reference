import sys
import argparse
from pathlib import Path
import pandas as pd
from code.inference import predict_from_csv


def main():
    parser = argparse.ArgumentParser(description="Pipeline for feature extraction and prediction")
    parser.add_argument(
        "--input",
        type=str,
        default="./raw.csv",
        help="Path to the raw CSV file (default: ./raw.csv)")

    args = parser.parse_args()

    print(f"Running prediction on: {args.input}")

    result = predict_from_csv(args.input)
    if not result:
        print("No results produced.")
        return

    # 转成 DataFrame，便于导出
    df = pd.DataFrame(result)

    cwd = Path().resolve()
    sif_dir = cwd / 'SIF_result'
    sgf_dir = cwd / 'SGF_result'
    sif_dir.mkdir(parents=True, exist_ok=True)
    sgf_dir.mkdir(parents=True, exist_ok=True)

    # 保存 CSV（每个任务一个表）
    sif_df = df[['SMILES', 'SIF']].copy()
    sgf_df = df[['SMILES', 'SGF']].copy()

    sif_csv = sif_dir / 'sif_results.csv'
    sgf_csv = sgf_dir / 'sgf_results.csv'
    sif_df.to_csv(sif_csv, index=False)
    sgf_df.to_csv(sgf_csv, index=False)

    # 也保存为简单的 HTML 表格，方便可视化查看
    sif_html = sif_dir / 'sif_results.html'
    sgf_html = sgf_dir / 'sgf_results.html'
    sif_df.to_html(sif_html, index=False)
    sgf_df.to_html(sgf_html, index=False)

    print(f"Saved SIF results: {sif_csv} and {sif_html}")
    print(f"Saved SGF results: {sgf_csv} and {sgf_html}")


if __name__ == "__main__":
    main()