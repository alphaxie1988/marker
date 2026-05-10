# Marker — Automated Document Grader

## What
Streamlit app for AI-powered grading of student answer scripts (`.docx` files).
- Stack: Python, Streamlit, python-docx, OpenAI (GPT-4o), Pandas
- Config: `config.json` defines papers, questions, rubrics, and weights

## Why
Automates marking of Biology/Chemistry exam papers by sending student text to GPT-4o with a rubric, returning structured scores.

## How
```bash
streamlit run app.py
pip install -r requirements.txt
```
Requires `OPENAI_API_KEY` in environment. Edit `config.json` to add papers or adjust rubrics.
