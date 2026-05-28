# Cognitia Chatbot
# By Raymond Hoogendoorn
# Copyright 2026

import json
from knowledge_claims_tool import (
    run_cimo_toulmin_analysis,
    run_cimo_structuring,
    run_semantic_network_builder,
    extract_concepts_from_document,
    build_toulmin_excel,
    build_toulmin_json,
    get_excel_extension,
)
import os
import pickle
from typing import Dict, List, Tuple

import urllib.request
import urllib.error

import numpy as np
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
try:
    from docx import Document
except Exception:
    Document = None

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3"
EMBEDDING_MODEL = "nomic-embed-text"

BASE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)

CACHE_DIR = os.path.join(BASE_DIR, ".rag_cache")
CACHE_META_PATH = os.path.join(CACHE_DIR, "meta.json")
CACHE_CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.pkl")
CACHE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "embeddings.npy")
CACHE_NORMS_PATH = os.path.join(CACHE_DIR, "norms.npy")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def _normalize_dir(path: str) -> str:
    """Normalize a directory path so it stays portable across environments."""
    return os.path.normpath(os.path.abspath(os.path.expanduser(path.strip())))


def _first_existing_dir(candidates: List[str]) -> str | None:
    for path in candidates:
        normalized = _normalize_dir(path)
        if os.path.isdir(normalized):
            return normalized
    return None


def _load_cached_roots() -> List[str] | None:
    """
    If an existing RAG cache is present, reuse its configured roots so we
    don't discard a valid cache when the code is moved to a new folder.
    """
    try:
        if os.path.isfile(CACHE_META_PATH):
            with open(CACHE_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            roots = meta.get("roots")
            if isinstance(roots, list) and roots:
                # Keep only existing directories; cached absolute paths from another
                # machine should not override local project-relative defaults.
                existing_roots = []
                for root in roots:
                    if not isinstance(root, str):
                        continue
                    normalized = _normalize_dir(root)
                    if os.path.isdir(normalized):
                        existing_roots.append(normalized)
                if existing_roots:
                    return existing_roots
    except Exception:
        # If anything goes wrong, just fall back to normal detection.
        pass
    return None


_cached_roots = _load_cached_roots()
if _cached_roots:
    PDF_ROOTS = _cached_roots
else:
    # Wisdom documents: support both project-local folder and shared folders one level up
    _wisdom_dir = _first_existing_dir(
        [
            os.path.join(BASE_DIR, "Wisdom documents"),
            os.path.join(PARENT_DIR, "Wisdom documents"),
            os.path.join(PARENT_DIR, "Wisdom documents - backup"),
        ]
    )
    if _wisdom_dir is None:
        _wisdom_dir = os.path.join(BASE_DIR, "Wisdom documents")

    PDF_ROOTS = [
        _wisdom_dir,
    ]


# Knowledge claims: support both project-local folder and shared folder one level up
_kc_dir = _first_existing_dir(
    [
        os.path.join(BASE_DIR, "Knowledge claims"),
        os.path.join(PARENT_DIR, "Knowledge claims"),
    ]
)
if _kc_dir is None:
    _kc_dir = os.path.join(BASE_DIR, "Knowledge claims")

KNOWLEDGE_CLAIMS_DIR = _kc_dir
# Hiërarchie uit map Knowledge claims (2025 11 13 kennisclaims_hierarchie.xlsx) indien aanwezig
DEFAULT_HIERARCHY_XLSX = os.path.join(KNOWLEDGE_CLAIMS_DIR, "2025 11 13 kennisclaims_hierarchie.xlsx")


def ollama_request(endpoint: str, payload: dict) -> dict:
    url = f"{OLLAMA_BASE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            response = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(f"Ollama error: {response['error']}")
    return response


def call_llm(messages, temperature: float = 0.4) -> str:
    response = ollama_request(
        "/api/chat",
        {
            "model": LLM_MODEL,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": False,
        },
    )
    return response.get("message", {}).get("content", "").strip()


def check_model_connections() -> Tuple[bool, str, bool, str]:
    """
    Checks Ollama reachability and availability of the LLM and embedding models.
    Returns:
    - llm_ok, llm_message
    - embedding_ok, embedding_message
    """
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        base_msg = (
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. Details: {exc}"
        )
        return (
            False,
            f"LLM connection failed. {base_msg}",
            False,
            f"Embedding connection failed. {base_msg}",
        )

    models = payload.get("models", []) if isinstance(payload, dict) else []
    available = {m.get("name", "").split(":")[0] for m in models if isinstance(m, dict)}

    llm_ok = LLM_MODEL in available
    embedding_ok = EMBEDDING_MODEL in available

    if llm_ok:
        llm_msg = "LLM connection is active."
    else:
        llm_msg = (
            f"LLM model '{LLM_MODEL}' is not installed. "
            f"Install with: ollama pull {LLM_MODEL}"
        )

    if embedding_ok:
        embedding_msg = "Embedding model connection is active."
    else:
        embedding_msg = (
            f"Embedding model '{EMBEDDING_MODEL}' is not installed. "
            f"Install with: ollama pull {EMBEDDING_MODEL}"
        )

    return llm_ok, llm_msg, embedding_ok, embedding_msg


def list_pdf_files(roots: List[str]) -> List[str]:
    pdfs: List[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for base, _, files in os.walk(root):
            for name in files:
                if name.lower().endswith(".pdf"):
                    pdfs.append(os.path.join(base, name))
    return pdfs


def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def get_pdf_metadata(path: str) -> Dict[str, str]:
    """Extracts title and author(s) from PDF metadata, if available."""
    try:
        reader = PdfReader(path)
        info = getattr(reader, "metadata", None) or getattr(reader, "documentInfo", None) or {}
        title = ""
        authors = ""
        if isinstance(info, dict):
            title = (info.get("/Title") or info.get("title") or "") or ""
            authors = (info.get("/Author") or info.get("author") or "") or ""
        else:
            title = getattr(info, "title", "") or ""
            authors = getattr(info, "author", "") or ""
        return {
            "title": str(title).strip(),
            "authors": str(authors).strip(),
        }
    except Exception:
        return {"title": "", "authors": ""}


def clean_text(text: str) -> str:
    return " ".join(text.replace("\t", " ").split())


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max(chunk_size - overlap, 1)
    return chunks


def embed_texts(
    texts: List[str],
    batch_size: int = 64,
    progress_bar: "st.delta_generator.DeltaGenerator | None" = None,
) -> np.ndarray:
    all_embeddings = []
    total = max(len(texts), 1)
    for idx, text in enumerate(texts, start=1):
        response = ollama_request(
            "/api/embeddings",
            {"model": EMBEDDING_MODEL, "prompt": text},
        )
        embedding = response.get("embedding", [])
        if not embedding:
            raise RuntimeError(
                "Embedding response is empty. Check that the embedding model is installed "
                f"(ollama pull {EMBEDDING_MODEL})."
            )
        all_embeddings.append(embedding)
        if progress_bar is not None:
            progress_bar.progress(
                idx / total,
                text=f"Creating embeddings... {idx}/{total}",
            )
    embeddings = np.array(all_embeddings, dtype=np.float32)
    if embeddings.ndim == 1 and embeddings.size > 0:
        embeddings = embeddings.reshape(1, -1)
    return embeddings


def get_file_signature(path: str) -> Dict[str, float]:
    stat = os.stat(path)
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def load_cache(
    roots: List[str],
) -> Tuple[List[dict], np.ndarray, np.ndarray, Dict[str, Dict[str, float]], bool]:
    if not os.path.exists(CACHE_META_PATH):
        return [], np.array([], dtype=np.float32), np.array([], dtype=np.float32), {}, False

    try:
        with open(CACHE_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if (
            meta.get("embedding_model") != EMBEDDING_MODEL
            or meta.get("chunk_size") != CHUNK_SIZE
            or meta.get("chunk_overlap") != CHUNK_OVERLAP
            or meta.get("roots") != roots
        ):
            return [], np.array([], dtype=np.float32), np.array([], dtype=np.float32), {}, False

        with open(CACHE_CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        embeddings = np.load(CACHE_EMBEDDINGS_PATH)
        norms = np.load(CACHE_NORMS_PATH)
        if embeddings.ndim == 1 and embeddings.size > 0:
            embeddings = embeddings.reshape(1, -1)
        if norms.ndim != 1 or norms.size != embeddings.shape[0]:
            norms = np.linalg.norm(embeddings, axis=1) if embeddings.size else np.array([], dtype=np.float32)
        return chunks, embeddings, norms, meta.get("files", {}), True
    except Exception:
        return [], np.array([], dtype=np.float32), np.array([], dtype=np.float32), {}, False


def save_cache(
    chunks: List[dict],
    embeddings: np.ndarray,
    norms: np.ndarray,
    files_meta: Dict[str, Dict[str, float]],
    roots: List[str],
) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    meta = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "files": files_meta,
        "roots": roots,
    }
    with open(CACHE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with open(CACHE_CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    np.save(CACHE_EMBEDDINGS_PATH, embeddings)
    np.save(CACHE_NORMS_PATH, norms)


def build_rag_index(roots: List[str]) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    normalized_roots = [_normalize_dir(r) for r in roots if isinstance(r, str) and r.strip()]
    if not normalized_roots:
        normalized_roots = roots

    pdf_paths = list_pdf_files(normalized_roots)
    current_files = {path: get_file_signature(path) for path in pdf_paths}

    cached_chunks, cached_embeddings, cached_norms, cached_files, cache_ok = load_cache(normalized_roots)

    removed_files = set(cached_files) - set(current_files)
    modified_files = [
        path for path in current_files if cached_files.get(path) != current_files[path]
    ]
    new_files = [path for path in current_files if path not in cached_files]

    rebuild_full = not cache_ok or removed_files or modified_files

    if rebuild_full:
        chunks: List[dict] = []
        progress = st.progress(0, text="Loading PDFs...")
        total_pdfs = max(len(pdf_paths), 1)

        for idx, path in enumerate(pdf_paths, start=1):
            meta = get_pdf_metadata(path)
            raw_text = extract_pdf_text(path)
            cleaned = clean_text(raw_text)
            for chunk in split_text(cleaned):
                chunks.append(
                    {
                        "text": chunk,
                        "source": os.path.basename(path),
                        "title": meta.get("title", ""),
                        "authors": meta.get("authors", ""),
                    }
                )
            progress.progress(idx / total_pdfs, text=f"Loading PDFs... {idx}/{total_pdfs}")

        if not chunks:
            progress.progress(1.0, text="No PDF files found.")
            save_cache(
                [],
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                current_files,
                normalized_roots,
            )
            return [], np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        progress.progress(1.0, text="Preparing embeddings...")
        progress_embeddings = st.progress(0, text="Creating embeddings...")
        try:
            embeddings = embed_texts([c["text"] for c in chunks], progress_bar=progress_embeddings)
        except RuntimeError as exc:
            st.error(str(exc))
            return [], np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        progress_embeddings.progress(1.0, text="Embeddings ready")
        norms = np.linalg.norm(embeddings, axis=1) if embeddings.size else np.array([], dtype=np.float32)
        save_cache(chunks, embeddings, norms, current_files, normalized_roots)
        return chunks, embeddings, norms

    if not new_files:
        return cached_chunks, cached_embeddings, cached_norms

    progress = st.progress(0, text="Loading new PDFs...")
    total_pdfs = max(len(new_files), 1)

    new_chunks: List[dict] = []
    for idx, path in enumerate(new_files, start=1):
        meta = get_pdf_metadata(path)
        raw_text = extract_pdf_text(path)
        cleaned = clean_text(raw_text)
        for chunk in split_text(cleaned):
            new_chunks.append(
                {
                    "text": chunk,
                    "source": os.path.basename(path),
                    "title": meta.get("title", ""),
                    "authors": meta.get("authors", ""),
                }
            )
        progress.progress(idx / total_pdfs, text=f"Loading new PDFs... {idx}/{total_pdfs}")

    if not new_chunks:
        save_cache(cached_chunks, cached_embeddings, cached_norms, current_files, normalized_roots)
        return cached_chunks, cached_embeddings, cached_norms

    progress.progress(1.0, text="Preparing embeddings...")
    progress_embeddings = st.progress(0, text="Creating embeddings...")
    try:
        new_embeddings = embed_texts([c["text"] for c in new_chunks], progress_bar=progress_embeddings)
    except RuntimeError as exc:
        st.error(str(exc))
        return cached_chunks, cached_embeddings, cached_norms
    progress_embeddings.progress(1.0, text="Embeddings ready")

    chunks = cached_chunks + new_chunks
    embeddings = np.vstack([cached_embeddings, new_embeddings])
    norms = np.linalg.norm(embeddings, axis=1) if embeddings.size else np.array([], dtype=np.float32)
    save_cache(chunks, embeddings, norms, current_files, normalized_roots)
    return chunks, embeddings, norms


def retrieve_context(
    query: str,
    chunks: List[dict],
    embeddings: np.ndarray,
    norms: np.ndarray,
    top_k: int = 4,
    min_score: float = 0.25,
) -> Tuple[str, List[str], float]:
    if not chunks or embeddings.size == 0 or norms.size == 0:
        return "", [], 0.0

    response = ollama_request(
        "/api/embeddings",
        {"model": EMBEDDING_MODEL, "prompt": query},
    )
    q_embedding = response.get("embedding", [])
    if not q_embedding:
        return "", [], 0.0
    q_vec = np.array(q_embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q_vec) + 1e-10

    scores = (embeddings @ q_vec) / (norms * q_norm + 1e-10)
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[top_indices]
    best_score = float(top_scores[0]) if len(top_scores) else 0.0

    if best_score < min_score:
        return "", [], best_score

    selected = [chunks[i] for i in top_indices]

    # Build source descriptions: PDF filename + (optional) title and author(s)
    by_source: Dict[str, Dict[str, str]] = {}
    for c in selected:
        src = c.get("source", "")
        if not src:
            continue
        if src not in by_source:
            by_source[src] = {
                "title": c.get("title", "") or "",
                "authors": c.get("authors", "") or "",
            }
    sources: List[str] = []
    for src, meta in by_source.items():
        desc = src
        if meta.get("title"):
            desc += f" — {meta['title']}"
        if meta.get("authors"):
            desc += f" (author(s): {meta['authors']})"
        sources.append(desc)

    context_parts = []
    for item in selected:
        src = item.get("source", "")
        title = item.get("title", "")
        authors = item.get("authors", "")
        header = src
        if title:
            header += f" — {title}"
        if authors:
            header += f" (author(s): {authors})"
        context_parts.append(f"[Source: {header}]\n{item['text']}")
    context = "\n\n".join(context_parts)
    return context, sources, best_score


def extract_docx_text(file_bytes: bytes) -> str:
    if Document is None:
        raise RuntimeError("python-docx is not installed. Run: python -m pip install python-docx")
    from io import BytesIO

    doc = Document(BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


st.set_page_config(page_title="Cognitia", page_icon="CG", layout="centered")
st.markdown(
    """
    <style>
        :root {
            --bg: #0b0f19;
            --surface: #121826;
            --surface-2: #0f1524;
            --text: #e5e7eb;
            --muted: #9ca3af;
            --accent: #3b82f6;
            --border: #1f2937;
        }
        .stApp {
            background-color: var(--bg);
            color: var(--text);
        }
        .block-container {
            background-color: var(--bg);
            color: var(--text);
            padding-top: 2rem;
        }
        section[data-testid="stSidebar"] {
            background-color: var(--surface-2);
            border-right: 1px solid var(--border);
        }
        .header-wrap {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.75rem 1rem;
            background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        }
        .header-title {
            font-size: 1.6rem;
            font-weight: 700;
            letter-spacing: 0.2px;
        }
        .header-subtitle {
            font-size: 0.95rem;
            color: var(--muted);
            margin-top: 0.1rem;
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        }
        .stChatMessage {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.3rem 0.8rem;
        }
        .stTextInput>div>div>input {
            background: var(--surface);
            color: var(--text);
            border: 1px solid var(--border);
        }
    </style>
    """,
    unsafe_allow_html=True,
)
logo_path = os.path.join(
    os.path.dirname(__file__),
    "ChatGPT Image Mar 11, 2026, 10_45_55 AM.png",
)
if os.path.isfile(logo_path):
    try:
        logo_image = Image.open(logo_path)
    except Exception:
        logo_image = None
else:
    logo_image = None

st.markdown('<div class="header-wrap">', unsafe_allow_html=True)
if logo_image is not None:
    st.image(logo_image, width=260)
st.markdown(
    """
    <div>
        <div class="header-subtitle">Scientific Knowledge Hub</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

llm_connected, llm_status_message, embedding_connected, embedding_status_message = check_model_connections()
if not llm_connected:
    st.error(
        "Cannot connect to the LLM right now. "
        "Please make sure Ollama is running and required models are installed."
    )
    st.caption(llm_status_message)
if not embedding_connected:
    st.error(
        "Cannot connect to the embedding model right now. "
        "Please make sure Ollama is running and required models are installed."
    )
    st.caption(embedding_status_message)

with st.sidebar:
    st.subheader("Documents")
    default_root = PDF_ROOTS[0]
    selected_root = st.text_input("Documents folder", value=default_root)
    selected_root = _normalize_dir(selected_root) if selected_root.strip() else _normalize_dir(default_root)
    st.caption(f"Folder exists: {os.path.isdir(selected_root)}")
    min_score = st.slider("Relevance threshold", 0.0, 0.5, 0.15, 0.01)
    mode = st.radio(
        "Mode",
        ["Chat", "Document review", "Knowledge claims"],
        index=0,
    )

with st.spinner("Loading and indexing documents..."):
    rag_chunks, rag_embeddings, rag_norms = build_rag_index([selected_root])

with st.sidebar:
    st.write(f"Chunks: {len(rag_chunks)}")
    if not rag_chunks:
        st.warning("No PDF files found in the selected folder.")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.caption("Models")
    st.write(f"LLM: {LLM_MODEL}")
    st.write(f"Embeddings: {EMBEDDING_MODEL}")
    if llm_connected:
        st.success("LLM status: connected")
    else:
        st.error("LLM status: not connected")
        st.caption(llm_status_message)
    if embedding_connected:
        st.success("Embedding status: connected")
    else:
        st.error("Embedding status: not connected")
        st.caption(embedding_status_message)
    st.markdown("</div>", unsafe_allow_html=True)

if mode == "Chat":
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        context, sources, score = retrieve_context(
            user_input, rag_chunks, rag_embeddings, rag_norms, min_score=min_score
        )

        if not context:
            assistant_reply = (
                "I cannot assess this based on the available documents."
            )
        else:
            system_prompt = (
                "You are Cognitia, a helpful chatbot for technical business administration. "
                "Communicate only in English. Use only the provided context. "
                "If the context is insufficient to answer, explicitly say: "
                "'I cannot assess this based on the available documents.'"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Context:\n{context}"},
            ]
            history = st.session_state.messages[-6:]
            messages.extend(history)

            try:
                assistant_reply = call_llm(messages)
            except Exception as exc:
                assistant_reply = f"Something went wrong with the LLM request: {exc}"

            if sources:
                sources_text = "\n".join([f"- {source}" for source in sources])
                assistant_reply = f"{assistant_reply}\n\nSources (PDF files with title and author(s)):\n{sources_text}"

        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

if mode == "Document review":
    st.subheader("Review a student document (Word)")
    st.caption("Upload a .docx file and click Generate review.")
    uploaded = st.file_uploader("Upload .docx", type=["docx"])
    generate = st.button("Generate review", type="primary", disabled=uploaded is None)

    if uploaded is not None:
        try:
            doc_text = extract_docx_text(uploaded.getvalue())
        except Exception as exc:
            st.error(str(exc))
            doc_text = ""

        if not doc_text:
            st.warning("No text extracted from the document.")
        else:
            preview = doc_text[:1200]
            with st.expander("Preview extracted text"):
                st.write(preview + ("..." if len(doc_text) > 1200 else ""))

            if generate:
                with st.spinner("Generating review..."):
                    context, sources, score = retrieve_context(
                        doc_text[:2000], rag_chunks, rag_embeddings, rag_norms, min_score=min_score
                    )

                    if not context:
                        st.warning("I cannot assess this based on the available documents.")
                    else:
                        review_prompt = (
                            "You are Cognitia, a strict but constructive reviewer. "
                            "Communicate only in English. Review the student's document using only the provided context. "
                            "Give: (1) a short summary, (2) key issues or gaps, "
                            "(3) specific improvement suggestions, and (4) a brief score out of 10. "
                            "If the context is insufficient, explicitly say: "
                            "'I cannot assess this based on the available documents.'"
                        )
                        messages = [
                            {"role": "system", "content": review_prompt},
                            {"role": "system", "content": f"Context:\n{context}"},
                            {"role": "user", "content": doc_text},
                        ]
                        try:
                            review_response = call_llm(messages)
                        except Exception as exc:
                            review_response = f"Something went wrong with the LLM request: {exc}"

                        st.markdown(review_response)
                        if sources:
                            st.markdown("**Sources (PDF files with title and author(s))**")
                            st.write("\n".join([f"- {source}" for source in sources]))

if mode == "Knowledge claims":
    def extract_upload_text(upload) -> str:
        if upload is None:
            return ""
        raw = upload.getvalue()
        name = upload.name or ""
        if name.lower().endswith(".pdf"):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(raw)
                tmp.flush()
                try:
                    return extract_pdf_text(tmp.name)
                finally:
                    try:
                        os.unlink(tmp.name)
                    except Exception:
                        pass
        if name.lower().endswith(".docx"):
            return extract_docx_text(raw)
        return ""

    st.subheader("Knowledge claims")
    st.caption(
        "Four-phase flow: (1) Aristoteles → semantic network, (2) Denyer → **generates CIMO conjectures**, "
        "(3) Toulmin-Weaver → analyses CIMO + knowledge levels, (4) Knowledge-Base Generator (later). "
        "You can generate CIMO with Denyer first, then feed it to Toulmin."
    )
    kc_tool = st.radio(
        "Choose a tool",
        [
            "Semantic network (Aristoteles)",
            "CIMO structuring (Denyer)",
            "CIMO analysis (Toulmin)",
        ],
        horizontal=True,
    )

    if kc_tool == "CIMO analysis (Toulmin)":
        st.markdown("**Analyse and qualify CIMO conjectures (Phase 3)**")
        st.caption(
            "Input: document with CIMO conjectures (upload or use generated CIMO from Denyer below). "
            "Optional: document(s) with data or evidence for support."
        )
        use_denyer_output = (
            st.checkbox(
                "Use generated CIMO conjectures from Denyer (Phase 2) as input",
                value=bool(st.session_state.get("denyer_generated_cimo")),
                key="toulmin_use_denyer",
            )
            and st.session_state.get("denyer_generated_cimo")
        )
        if use_denyer_output and st.session_state.get("denyer_generated_cimo"):
            with st.expander("View generated CIMO conjectures (from Denyer)"):
                preview = st.session_state["denyer_generated_cimo"]
                st.text(preview[:4000] + ("…" if len(preview) > 4000 else ""))
                if len(preview) > 4000:
                    st.caption("(first 4000 characters; full text is used for analysis)")
        doc_upload = st.file_uploader(
            "Document with CIMO conjectures (PDF or Word)",
            type=["pdf", "docx"],
            key="toulmin_doc",
        )
        evidence_uploads = st.file_uploader(
            "Optional: document(s) with data/evidence (PDF or Word)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="toulmin_evidence",
        )
        hierarchy_upload = st.file_uploader(
            "Optional: knowledge-claims hierarchy (Excel .xlsx) – for more accurate knowledge level",
            type=["xlsx"],
            key="toulmin_hierarchy",
        )
        if os.path.isfile(DEFAULT_HIERARCHY_XLSX):
            st.caption(f"Default hierarchy from Knowledge claims folder will be used: 2025 11 13 kennisclaims_hierarchie.xlsx")
        else:
            st.caption("Without a hierarchy file, the built-in 10-level hierarchy (PDF table) is used. Place an .xlsx in the Knowledge claims folder to use it.")
        use_wisdom = st.checkbox(
            "Also determine certainty level using knowledge from Cognitia (document base)",
            value=True,
            key="toulmin_use_wisdom",
        )
        st.caption("When checked, relevant literature from the selected documents folder is included to support the knowledge level.")
        run_toulmin = st.button("Start CIMO and Toulmin analysis", type="primary")

        if run_toulmin and (doc_upload or use_denyer_output):
            conjectures_text = (
                (st.session_state.get("denyer_generated_cimo") or "").strip()
                if use_denyer_output
                else extract_upload_text(doc_upload)
            )
            if not conjectures_text.strip():
                st.warning("No text extracted from the document.")
            else:
                evidence_texts = []
                for up in evidence_uploads or []:
                    t = extract_upload_text(up)
                    if t.strip():
                        evidence_texts.append(t[:8000])
                conjectures_trimmed = conjectures_text[:15000]
                hierarchy_bytes = hierarchy_upload.getvalue() if hierarchy_upload else None
                hierarchy_path = None
                if not hierarchy_bytes and os.path.isfile(DEFAULT_HIERARCHY_XLSX):
                    hierarchy_path = DEFAULT_HIERARCHY_XLSX

                wisdom_context = ""
                wisdom_sources: List[str] = []
                if use_wisdom and rag_chunks and rag_embeddings.size > 0:
                    with st.spinner("Searching Cognitia documents..."):
                        wisdom_context, wisdom_sources, _ = retrieve_context(
                            conjectures_trimmed[:3000],
                            rag_chunks,
                            rag_embeddings,
                            rag_norms,
                            top_k=6,
                            min_score=min_score,
                        )
                    if wisdom_sources:
                        st.caption(f"Cognitia: {len(wisdom_sources)} source(s) (PDF + title + author(s)) included for certainty level:")
                        st.write("\n".join([f"- {src}" for src in wisdom_sources]))

                with st.spinner("Running CIMO and Toulmin analysis..."):
                    try:
                        result = run_cimo_toulmin_analysis(
                            conjectures_trimmed,
                            evidence_texts if evidence_texts else None,
                            call_llm=lambda msgs, **kw: call_llm(msgs, **kw),
                            hierarchy_xlsx_bytes=hierarchy_bytes,
                            hierarchy_xlsx_path=hierarchy_path,
                            wisdom_context=wisdom_context if wisdom_context else None,
                        )
                    except Exception as exc:
                        result = f"Something went wrong: {exc}"
                st.markdown(result)
                st.session_state["toulmin_last_result"] = result
        if st.session_state.get("toulmin_last_result"):
            st.markdown("---")
            st.caption("Export (technical spec: Excel and/or JSON)")
            excel_bytes = build_toulmin_excel(st.session_state["toulmin_last_result"])
            ext = get_excel_extension()
            st.download_button(
                f"Download as {ext}",
                data=excel_bytes,
                file_name=f"cimo_toulmin_analyse{ext}",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if ext == ".xlsx" else "text/csv",
                key="dl_excel_toulmin",
            )
            json_str = build_toulmin_json(st.session_state["toulmin_last_result"])
            st.download_button(
                "Download as JSON (evidence_matrix)",
                data=json_str,
                file_name="cimo_toulmin_evidence_matrix.json",
                mime="application/json",
                key="dl_json_toulmin",
            )

    elif kc_tool == "CIMO structuring (Denyer)":
        st.markdown("**Phase 2: Generate CIMO conjectures (Denyer)**")
        st.caption(
            "**Input for CIMO structuring:** a semantic network — i.e. a domain description with concepts, "
            "relations and influences. Source: (1) output from Aristoteles (Phase 1), (2) your own text, or (3) an uploaded PDF/Word. "
            "Denyer generates CIMO rows and conjectures from it. You can then feed them to Toulmin (Phase 3)."
        )
        use_aristoteles_output = (
            st.checkbox(
                "Use semantic network from Aristoteles (Phase 1) as input",
                value=bool(st.session_state.get("aristoteles_semantic_network")),
                key="denyer_use_aristoteles",
            )
            and st.session_state.get("aristoteles_semantic_network")
        )
        if use_aristoteles_output and st.session_state.get("aristoteles_semantic_network"):
            denyer_input = st.session_state["aristoteles_semantic_network"]
            with st.expander("View semantic network used (from Aristoteles)"):
                preview = denyer_input[:3000] + ("…" if len(denyer_input) > 3000 else "")
                st.text(preview)
        else:
            denyer_input = st.text_area(
                "Semantic network (text)",
                height=200,
                placeholder="Paste the semantic network text here (e.g. output from Aristoteles, or your own domain description with concepts and relations)...",
                key="denyer_text",
            )
            denyer_file = st.file_uploader(
                "Or upload a file (PDF/Word) with domain description or semantic network",
                type=["pdf", "docx"],
                key="denyer_upload",
            )
            if denyer_file:
                denyer_input = extract_upload_text(denyer_file) or denyer_input
        run_denyer = st.button("Start CIMO structuring", type="primary")
        if run_denyer and (denyer_input or "").strip():
            with st.spinner("Generating CIMO conjectures..."):
                try:
                    denyer_result = run_cimo_structuring(
                        (denyer_input or "")[:20000],
                        call_llm=lambda msgs, **kw: call_llm(msgs, **kw),
                    )
                except Exception as exc:
                    denyer_result = f"Something went wrong: {exc}"
            st.markdown(denyer_result)
            st.session_state["denyer_generated_cimo"] = denyer_result
            st.success("CIMO conjectures generated. Go to **CIMO analysis (Toulmin)** and check 'Use generated CIMO conjectures from Denyer' to analyse them.")
        elif run_denyer:
            st.warning("Enter text or upload a file first.")

    else:
        st.markdown("**Semantic network around a central concept (Aristoteles)**")
        st.caption(
            "Choose manual input or upload a Word/PDF: the tool will extract central and related concepts from the document and build the semantic network (expansion to 25 concepts, definitions, relations, hierarchy)."
        )
        arist_input_mode = st.radio(
            "Enter concepts",
            ["Manual: fill in the fields below", "From document: upload Word or PDF"],
            key="arist_input_mode",
            horizontal=True,
        )

        central = ""
        related_list = []
        doc_text_arist = ""

        if arist_input_mode == "From document: upload Word or PDF":
            arist_upload = st.file_uploader(
                "Upload Word (.docx) or PDF",
                type=["docx", "pdf"],
                key="arist_doc_upload",
            )
            if arist_upload is not None:
                with st.spinner("Reading document..."):
                    doc_text_arist = extract_upload_text(arist_upload)
                if not (doc_text_arist or "").strip():
                    st.warning("No text extracted from the document. Try another file.")
                else:
                    st.session_state["arist_uploaded_doc_text"] = doc_text_arist
                    with st.expander("Preview document text"):
                        st.text(doc_text_arist[:2000] + ("..." if len(doc_text_arist) > 2000 else ""))
        else:
            central = (st.text_input("Central concept", placeholder="e.g. Sustainable business", key="arist_central") or "").strip()
            related_raw = st.text_input(
                "Related concepts (max 10, comma-separated)",
                placeholder="e.g. circular economy, CSR, supply chain management",
                key="arist_related",
            )
            related_list = [x.strip() for x in (related_raw or "").split(",") if x.strip()][:10]

        run_arist = st.button("Generate semantic network", type="primary")

        if run_arist:
            if arist_input_mode == "From document: upload Word or PDF":
                # Use session state so we still have the document after button click (uploader resets on rerun)
                doc_to_use = (doc_text_arist or "").strip() or (st.session_state.get("arist_uploaded_doc_text") or "")
                if not doc_to_use:
                    st.warning("Upload a Word or PDF document first, or the document could not be read.")
                else:
                    with st.spinner("Extracting concepts from document..."):
                        try:
                            central, related_list = extract_concepts_from_document(
                                doc_to_use,
                                call_llm=lambda msgs, **kw: call_llm(msgs, **kw),
                            )
                        except Exception as exc:
                            st.error(f"Concept extraction failed: {exc}")
                            central, related_list = "", []
                    if central or related_list:
                        st.info(f"Found: **central** = {central or '(none)'}, **related** = {', '.join(related_list) or '(none)'}. Building semantic network.")
                    if (central or "").strip() or related_list:
                        with st.spinner("Building semantic network..."):
                            try:
                                if not (central or "").strip() and related_list:
                                    central = related_list[0]
                                    related_list = related_list[1:10]
                                arist_result = run_semantic_network_builder(
                                    (central or "").strip() or "Main concept",
                                    related_list,
                                    call_llm=lambda msgs, **kw: call_llm(msgs, **kw),
                                )
                            except Exception as exc:
                                arist_result = f"Something went wrong: {exc}"
                        st.markdown(arist_result)
                        st.session_state["aristoteles_semantic_network"] = arist_result
                        st.success("Semantic network generated. Go to **CIMO structuring (Denyer)** and check 'Use semantic network from Aristoteles' to generate CIMO conjectures.")
                    elif not (central or related_list):
                        st.warning("No concepts extracted from the document. Try another document or manual input.")
            else:
                if not (central or "").strip():
                    st.warning("Enter at least the central concept.")
                else:
                    with st.spinner("Building semantic network..."):
                        try:
                            arist_result = run_semantic_network_builder(
                                central,
                                related_list,
                                call_llm=lambda msgs, **kw: call_llm(msgs, **kw),
                            )
                        except Exception as exc:
                            arist_result = f"Something went wrong: {exc}"
                    st.markdown(arist_result)
                    st.session_state["aristoteles_semantic_network"] = arist_result
                    st.success("Semantic network generated. Go to **CIMO structuring (Denyer)** and check 'Use semantic network from Aristoteles' to generate CIMO conjectures.")

