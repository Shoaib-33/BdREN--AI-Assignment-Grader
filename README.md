# BdREN AI Assignment Grader

A Streamlit app that grades ML assignment answers against the supplied rubric and course-book excerpt.

## What it does

- Extracts text from the course book PDF.
- Embeds book chunks with a HuggingFace sentence-transformer model.
- Stores and searches the chunks in a persistent Chroma vector database.
- Retrieves book evidence separately for every rubric criterion.
- Uses an OpenAI-compatible grading model to produce rubric scores, feedback, book references, and unsupported-claim flags.
- Runs a second model checker pass to verify arithmetic, rubric bounds, grounding, and flags.
- Saves Markdown reports in `reports/` and an audit log in `grading_log.jsonl`.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Add your grading model key to `.env`:

```text
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL=gpt-4o-mini
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

`HF_EMBEDDING_MODEL` is optional. The default is `sentence-transformers/all-MiniLM-L6-v2`.

## Reports

Use the sidebar button **Grade all sample submissions** after setting `OPENAI_API_KEY`. The app writes:

- `reports/student_A_report.md`
- `reports/student_B_report.md`
- `reports/student_C_report.md`
