import argparse
import json
import os

import pandas as pd

from app.llm_client import LLMClient
from app.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run short video data analysis pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--target", "-t", required=True, help="Target metric column, e.g. sales")
    parser.add_argument("--output", "-o", default="report.json", help="Output JSON path")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required. Put it in environment or .env")

    llm = LLMClient(api_key=api_key, model=os.getenv("OPENAI_MODEL", "gpt-4"))
    df = pd.read_csv(args.input)
    report = run_pipeline(df, args.target, llm)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
