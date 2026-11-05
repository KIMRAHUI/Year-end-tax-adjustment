"""PDF → 챗봇용 검색 데이터(chunks.jsonl, emb.npy)를 만든다. (Colab 없이 내 컴퓨터에서 실행)

노트북 4~7절과 같은 방식이다.
  - PyMuPDF로 쪽별 텍스트를 읽고, 머리글·쪽번호 줄을 지운 뒤 NFC 정규화
  - 짧은 쪽(표지·간지, 30자 미만)은 제외
  - RecursiveCharacterTextSplitter(size 500, overlap 50), chunk_id = p216-c01
  - 임베딩: BAAI/bge-m3 (정규화), 결과는 float16으로 저장

실행:  python make_data.py "C:\\경로\\2024년 연말정산 신고안내.pdf"
결과:  data/chunks.jsonl, data/emb.npy  (같은 폴더의 data/에 만든다)
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import numpy as np

PAGE_OFFSET = 16
MIN_PAGE_CHARS = 30
CHUNK_SIZE, CHUNK_OVERLAP = 500, 50
DROP_LINE_PATTERNS = (r"^\s*원천징수의무자를\s*위한\s*$", r"^\s*2024년\s*연말정산\s*신고안내\s*$", r"^\s*\d{1,3}\s*$")
EMBED_MODEL = "BAAI/bge-m3"


class Page(NamedTuple):
    pdf_page: int
    text: str


class Chunk(NamedTuple):
    chunk_id: str
    pdf_page: int
    text: str


def load_pages(pdf_path: Path) -> tuple[list[Page], int]:
    """쪽별 텍스트를 읽어 정제한다. (사용할 쪽 목록, 전체 쪽 수)를 돌려준다."""
    import pymupdf
    drop = [re.compile(p) for p in DROP_LINE_PATTERNS]
    pages: list[Page] = []
    with pymupdf.open(pdf_path) as doc:
        total = len(doc)
        for i, page in enumerate(doc, start=1):
            text = unicodedata.normalize("NFC", page.get_text())
            kept = [ln for ln in text.splitlines() if not any(r.match(ln) for r in drop)]
            text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
            if len(text) >= MIN_PAGE_CHARS:
                pages.append(Page(i, text))
    return pages, total


def split_pages(pages: Sequence[Page]) -> list[Chunk]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                              separators=["\n\n", "\n", " ", ""], length_function=len)
    counter: Counter[int] = Counter()
    chunks: list[Chunk] = []
    for p in pages:
        for piece in splitter.split_text(p.text):
            counter[p.pdf_page] += 1
            chunks.append(Chunk(f"p{p.pdf_page:03d}-c{counter[p.pdf_page]:02d}", p.pdf_page, piece))
    return chunks


def embed_with_bge_m3(texts: Sequence[str]) -> np.ndarray:
    """bge-m3로 문서 벡터를 만든다 (CPU에서 5~15분, 처음 한 번은 모델 2GB를 내려받는다)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    return np.asarray(model.encode(list(texts), batch_size=8, normalize_embeddings=True, show_progress_bar=True))


def write_outputs(chunks: Sequence[Chunk], vectors: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"chunk_id": c.chunk_id, "pdf_page": c.pdf_page, "text": c.text}, ensure_ascii=False) + "\n")
    np.save(out_dir / "emb.npy", vectors.astype(np.float16))


def main(pdf: Path, out_dir: Path, embed: Callable[[Sequence[str]], np.ndarray] = embed_with_bge_m3) -> None:
    pages, total = load_pages(pdf)
    print(f"전체 {total}쪽 중 사용 {len(pages)}쪽  (노트북 기준: 426쪽 중 397쪽)")
    chunks = split_pages(pages)
    print(f"청크 {len(chunks)}개  (노트북 기준: 약 1,000개)")
    vectors = embed([c.text for c in chunks])
    assert vectors.shape[0] == len(chunks)
    write_outputs(chunks, vectors, out_dir)
    for p in sorted(out_dir.iterdir()):
        print(f"{p.name}: {p.stat().st_size / 1e6:.1f}MB")
    print("완료:", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path, help="연말정산 신고안내 PDF 경로")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    args = ap.parse_args()
    main(args.pdf, args.out)
