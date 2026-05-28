# Cognitia 2

Cognitia 2 is a local Streamlit app for evidence-based analysis on academic and professional documents, powered by local Ollama models.

It combines:
- document-grounded chat (RAG over PDF collections),
- structured document review for Word files, and
- a multi-phase knowledge-claims workflow (Aristoteles -> Denyer -> Toulmin).

## Main Features

- **RAG Chat mode**
  - Ask questions against your PDF knowledge base.
  - Answers are grounded in retrieved context and include source references.

- **Document review mode**
  - Upload a `.docx` student or working document.
  - Generate a structured review with summary, gaps, suggestions, and a score.

- **Knowledge claims mode**
  - Build semantic networks from manual input or uploaded documents.
  - Generate CIMO conjectures from semantic networks.
  - Run Toulmin-style CIMO analysis with optional evidence files and hierarchy Excel input.
  - Export Toulmin results as Excel/CSV and JSON.

## Tech Stack

- Python
- Streamlit
- Ollama (local LLM + embedding endpoints)
- NumPy
- PyPDF2
- python-docx
- Pillow
- openpyxl

## Requirements

- Python 3.10+
- Ollama installed and running locally at `http://localhost:11434`
- Ollama models:
  - `llama3`
  - `nomic-embed-text`

## Quick Start (Windows / PowerShell)

1. Open a terminal in this project folder:
   ```powershell
   cd "C:\Users\Raymo\OneDrive\Documents\Python Scripts\Cognitia_2"
   ```

2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Pull required Ollama models:
   ```powershell
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

5. Start the app:
   ```powershell
   streamlit run cognitia.py
   ```
   If `streamlit` is not recognized:
   ```powershell
   python -m streamlit run cognitia.py
   ```

6. Open:
   - `http://localhost:8501`

## Project Structure

- `cognitia.py` - Main Streamlit application
- `knowledge_claims_tool.py` - Knowledge-claims processing helpers
- `requirements.txt` - Python dependencies
- `INSTALLATIE.md` - Detailed Dutch installation guide
- `.rag_cache/` - Cached chunks and embeddings for faster RAG indexing
- `Wisdom documents/` - Default folder for PDF knowledge documents
- `Knowledge claims/` - Optional hierarchy and claims-related files

## Notes

- The app expects local Ollama connectivity and will show status in the sidebar.
- If no hierarchy Excel file is available, the Toulmin flow falls back to a built-in hierarchy.
- RAG indexing uses a local cache in `.rag_cache` and refreshes when source files change.

## Troubleshooting

For detailed troubleshooting (model connection, missing modules, Streamlit command issues), see `INSTALLATIE.md`.
