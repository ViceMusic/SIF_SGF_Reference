import sys
import argparse
from code.extract import test, process_by_raw_csv
from code.predict import test1, predict_by_processed_data

def main():
    # 1. 创建参数解析器
    parser = argparse.ArgumentParser(description="Pipeline for feature extraction and prediction")
    
    # 2. 添加路径参数。default 设置为你的原路径，这样不输入时也能跑
    parser.add_argument(
        "--input", 
        type=str, 
        default="./raw.csv", 
        help="Path to the raw CSV file (default: ./raw.csv)"
    )

    # 3. 解析参数
    args = parser.parse_args()

    print(f"Hello from feature-extraction! Using input: {args.input}")
    
    # 4. 调用函数，传入解析到的路径
    process_by_raw_csv(args.input)
    predict_by_processed_data()

if __name__ == "__main__":
    main()