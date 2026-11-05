# 연말정산 RAG 에이전트 챗봇 (Streamlit)

국세청 「2024년 연말정산 신고안내」를 근거로 답하는 챗봇입니다. (미션 14: RAG 에이전트)

| 구성 | 내용 |
|---|---|
| LLM | Groq `gpt-oss-120b` / `gpt-oss-20b` (무료 API) |
| 검색 | BM25(글자 2-gram) + bge-m3 dense → RRF |
| 에이전트 | LangChain `create_agent` + 문서 검색 도구 + 세액 계산 도구 4종 + 가드레일 |
| 기억 | 같은 세션 안에서 이전 질문을 기억 (`InMemorySaver`) |

## 배포 순서
1. 검색 데이터를 만듭니다. (둘 중 하나)
   - **내 컴퓨터(VS Code)**: `pip install -r requirements-data.txt` 후 `python make_data.py "PDF경로"` → `data/chunks.jsonl`, `data/emb.npy` 생성
   - Colab: `export_cell.py`의 셀을 노트북에서 실행한 뒤 두 파일을 `data/`에 넣습니다.
2. 이 폴더 전체를 GitHub 저장소에 올립니다. **키가 든 파일은 올리지 않습니다.** (`.streamlit/secrets.toml`은 `.gitignore`에 있습니다)
3. https://share.streamlit.io 에서 저장소를 연결하고 메인 파일을 `app.py`로 지정합니다.
4. 앱 설정 → Secrets에 `.streamlit/secrets.toml.example`의 형식으로 `GROQ_API_KEY`(와 선택으로 `APP_PASSWORD`)를 넣습니다.

## 주의
- 무료 Groq 한도(모델당 하루 20만 토큰)를 방문자가 함께 씁니다. 세션당 질문 수를 제한하고, 필요하면 접속 암호를 두세요.
- 메모리가 모자라 dense 모델을 못 올리면 앱이 BM25만으로 계속 동작합니다. 처음부터 BM25만 쓰려면 Secrets에 `RETRIEVAL_MODE = "bm25"`를 넣습니다.
- 챗봇의 답은 참고용입니다. 실제 신고 전에는 국세청 안내를 확인하세요.
