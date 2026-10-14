import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import pandas as pd

from app.pipeline import run_pipeline
from app.llm_client import LLMClient

app = FastAPI(title="Short Video Agent")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(file: UploadFile = File(...), target: str = Form(...)):
    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    llm = LLMClient(api_key=api_key, model=os.getenv("OPENAI_MODEL", "gpt-4"))
    report = run_pipeline(df, target, llm)
    return report
