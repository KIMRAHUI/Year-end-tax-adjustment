# ▶ Colab 노트북의 맨 아래(또는 7절 이후)에 새 셀로 넣고 실행하세요.
#   검색에 쓰는 청크 텍스트와 문서 벡터를 파일 2개로 내보냅니다. (CHUNKS, EMBEDDINGS는 노트북의 7절 변수)
import json
from pathlib import Path

import numpy as np

out_dir = CFG.paths.out_dir / "streamlit_data"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
    for c in CHUNKS:
        f.write(json.dumps({"chunk_id": str(c.metadata["chunk_id"]), "pdf_page": int(c.metadata["pdf_page"]),
                            "text": c.page_content}, ensure_ascii=False) + "\n")

vecs = np.asarray(EMBEDDINGS.embed_documents([c.page_content for c in CHUNKS]), dtype=np.float16)
np.save(out_dir / "emb.npy", vecs)

print("청크", len(CHUNKS), "개 / 벡터", vecs.shape)
for p in sorted(out_dir.iterdir()):
    print(f"{p.name}: {p.stat().st_size / 1e6:.1f}MB")
print("저장 위치:", out_dir)
