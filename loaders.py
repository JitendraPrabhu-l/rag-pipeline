"""Multi-format document loaders for the RAG pipeline.

Each loader turns a source file into one or more raw text chunks. `load_documents`
walks a file or directory and returns a flat list of {"id", "text", "source"}
records compatible with the `documents` list expected by rag.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _load_txt(path: Path) -> list[str]:
    return [path.read_text(encoding="utf-8", errors="replace")]


def _load_md(path: Path) -> list[str]:
    import markdown
    from bs4 import BeautifulSoup

    raw = path.read_text(encoding="utf-8", errors="replace")
    html = markdown.markdown(raw)
    return [BeautifulSoup(html, "html.parser").get_text()]


def _load_html(path: Path) -> list[str]:
    from bs4 import BeautifulSoup

    raw = path.read_text(encoding="utf-8", errors="replace")
    return [BeautifulSoup(raw, "html.parser").get_text()]


def _load_csv(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    return [" | ".join(f"{col}: {val}" for col, val in row.items()) for _, row in frame.iterrows()]


def _load_excel(path: Path) -> list[str]:
    frame = pd.read_excel(path)
    return [" | ".join(f"{col}: {val}" for col, val in row.items()) for _, row in frame.iterrows()]


def _load_json(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        texts = []
        for item in data:
            if isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
            else:
                texts.append(json.dumps(item, ensure_ascii=False))
        return texts
    return [json.dumps(data, ensure_ascii=False)]


def _load_pdf(path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def _load_docx(path: Path) -> list[str]:
    import docx

    document = docx.Document(str(path))
    return ["\n".join(p.text for p in document.paragraphs if p.text.strip())]


def _load_doc(path: Path) -> list[str]:
    try:
        return _load_docx(path)
    except Exception as exc:
        raise ValueError(
            f"cannot read legacy .doc file: no .doc parser is installed. "
            "Convert it to .docx or install a library such as 'textract'."
        ) from exc


_LOADERS = {
    ".txt": _load_txt,
    ".md": _load_md,
    ".markdown": _load_md,
    ".html": _load_html,
    ".htm": _load_html,
    ".csv": _load_csv,
    ".xlsx": _load_excel,
    ".xls": _load_excel,
    ".json": _load_json,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".doc": _load_doc,
}


def load_file(path: str | Path) -> list[dict]:
    """Load a single file into one or more {"text", "source"} chunks (e.g. one per PDF page/CSV row)."""
    path = Path(path)
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"unsupported file type: {path.suffix or '(none)'}")

    chunks = [text.strip() for text in loader(path) if text and text.strip()]
    return [{"text": text, "source": path.name} for text in chunks]


def load_documents(path: str | Path) -> list[dict]:
    """Load a single file, or every supported file under a directory, into a documents list with sequential ids."""
    path = Path(path)
    files = (
        [path]
        if path.is_file()
        else sorted(p for p in path.rglob("*") if p.suffix.lower() in _LOADERS)
    )

    records = []
    for file_path in files:
        try:
            records.extend(load_file(file_path))
        except Exception as exc:
            print(f"[WARN] Skipping '{file_path}': {exc}")

    return [{"id": idx + 1, **record} for idx, record in enumerate(records)]
