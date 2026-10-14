# Short Video Agent

A lightweight short video data analysis Agent, built around the closed loop of "Data Cleaning -> Attribution Analysis -> LLM Interpretation -> Strategy Output". It is suitable for business scenarios such as short video platforms, social media advertising, and content operations.

## Project Goals

- Automatically clean short video data (metrics such as impressions, clicks, views, interactions, conversions, etc.)
- Identify viral elements and weaknesses (title, cover, duration, publish time, content type, etc.)
- Use traditional machine learning to calculate feature importance
- Use LLMs to output Chinese attribution explanations and optimization strategies
- Provide HTTP API and command-line entry points for easy development and deployment

## Directory Structure

- app/cleaner.py: Data cleaning and feature engineering
- app/attribution.py: Traditional attribution capabilities (Random Forest + permutation importance)
- app/llm_client.py: LLM API wrapper
- app/pipeline.py: Main pipeline orchestration
- app/main.py: FastAPI endpoints
- run_pipeline.py: Command-line entry point
- sample_data/products_sample.csv: Sample data
- tests/test_pipeline.py: Basic tests

## Quick Start

1. Create a virtual environment

python -m venv .venv
source .venv/bin/activate

2. Install dependencies

pip install -r requirements.txt

3. Set environment variables

cp .env.example .env

Edit .env:

OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4

4. Run command-line example

python run_pipeline.py --input sample_data/products_sample.csv --target sales --output report.json

5. Start HTTP service

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Call the API:

curl -F "file=@sample_data/products_sample.csv" -F "target=sales" http://127.0.0.1:8000/analyze

## Output Explanation

The analysis result JSON will contain:
- clean_summary: Overview of cleaned data
- top_features: Ranking of important features from traditional models
- llm_explanation: Chinese attribution and strategies output by the LLM

## Example Business Questions

- Which content features are most likely to produce viral hits?
- Which weak points cause a drop in conversion?
- Which metrics should this video/account prioritize improving?
- What adjustments should be made next in terms of title, cover, duration, publish time, and content direction?

## Tech Stack

- Python 3.11+
- FastAPI
- Pandas / NumPy
- scikit-learn
- OpenAI API
- Docker (Optional)

## Notes

- LLM attribution relies on an API Key. It is recommended to customize the prompt based on business data.
- The current implementation aims for "lightweight, actionable, and rapid validation," suitable for product prototypes, data analysis scripts, and internal tool scenarios.
- If integrating with real production traffic, it is recommended to add SHAP, experimental design, dimensional stability checks, and more granular feature engineering.
