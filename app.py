"""연말정산 RAG 에이전트 챗봇 (Streamlit).

- 문서: 국세청 「2024년 연말정산 신고안내」
- LLM: Groq 무료 API (키는 Streamlit Secrets의 GROQ_API_KEY)
- 검색: BM25 + bge-m3 dense (RRF), 도구: 문서 검색 + 세액 계산 4종
"""
from __future__ import annotations

import os
import uuid
from typing import Any

import numpy as np
import streamlit as st

import rag_core as rc

st.set_page_config(page_title="연말정산 도우미", page_icon="🧾", layout="centered")

MODELS = {"gpt-oss-120b (정확, 기본)": "openai/gpt-oss-120b", "gpt-oss-20b (가벼움)": "openai/gpt-oss-20b"}
MAX_QUESTIONS = int(st.secrets.get("MAX_QUESTIONS_PER_SESSION", 12))
EXAMPLES = ["2024년 개정 세법에서 월세액 세액공제는 어떻게 바뀌었어?",
            "총급여 5,000만원인 무주택 세대주가 월 60만원씩 12개월 월세를 냈어. 월세 세액공제액은?",
            "자녀가 3명이면 2024년 자녀세액공제액은?",
            "암호화폐 거래로 번 수익에 대한 소득세율은?"]


# ───────── 접근 제어와 키 ─────────
def require_password() -> None:
    """APP_PASSWORD가 설정돼 있으면 입력해야 쓸 수 있다 (공용 Groq 한도를 지키기 위함)."""
    expected = st.secrets.get("APP_PASSWORD", "")
    if not expected or st.session_state.get("authed"):
        return
    pw = st.text_input("접속 암호", type="password")
    if pw and pw == expected:
        st.session_state["authed"] = True
        st.rerun()
    elif pw:
        st.error("암호가 맞지 않습니다.")
    st.stop()


def load_api_key() -> None:
    key = st.secrets.get("GROQ_API_KEY", "")
    if not key:
        st.error("Groq API 키가 설정되지 않았습니다. Streamlit의 앱 설정 → Secrets에 `GROQ_API_KEY`를 넣어 주세요.")
        st.stop()
    os.environ["GROQ_API_KEY"] = key


# ───────── 무거운 객체는 한 번만 만든다 ─────────
@st.cache_resource(show_spinner="검색 색인을 불러오는 중...")
def get_retriever() -> rc.Retriever:
    chunks = rc.load_chunks()
    vec_path = rc.DATA_DIR / "emb.npy"
    if st.secrets.get("RETRIEVAL_MODE", "auto") == "bm25" or not vec_path.exists():
        return rc.Retriever(chunks)
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(rc.EMBED_MODEL, device="cpu", model_kwargs={"torch_dtype": torch.float16})

        def embed(q: str) -> np.ndarray:
            return np.asarray(model.encode([q], normalize_embeddings=True)[0], dtype=np.float32)

        return rc.Retriever(chunks, np.load(vec_path), embed)
    except Exception as e:                      # 메모리 부족 등 → BM25만으로 계속 동작
        st.session_state["retriever_warning"] = f"dense 모델을 불러오지 못해 BM25로 동작합니다 ({type(e).__name__})"
        return rc.Retriever(chunks)


def get_agent(model_id: str) -> tuple[Any, Any, rc.ToolLog]:
    """세션마다 (에이전트, LLM, 도구 기록)을 하나씩 가진다."""
    key = f"agent::{model_id}"
    if key not in st.session_state:
        log = rc.ToolLog()
        llm = rc.make_llm(model_id)
        st.session_state[key] = (rc.build_agent(llm, get_retriever(), log), llm, log)
    return st.session_state[key]


# ───────── 화면 ─────────
st.title("🧾 연말정산 도우미")
st.caption("국세청 「2024년 연말정산 신고안내」를 근거로 답합니다. 참고용이며 실제 신고는 국세청 안내를 확인하세요.")
require_password()
load_api_key()

with st.sidebar:
    st.header("설정")
    model_label = st.selectbox("모델", list(MODELS))
    model_id = MODELS[model_label]
    retriever = get_retriever()
    st.write(f"검색 방식: **{retriever.mode}**")
    if "retriever_warning" in st.session_state:
        st.warning(st.session_state["retriever_warning"])
    quota_slot = st.empty()   # 질문을 처리한 뒤에 다시 채운다 (사이드바는 화면 위쪽이라 먼저 그려지기 때문)
    if st.button("대화 지우기"):
        for k in [k for k in st.session_state if k.startswith("agent::")] + ["messages", "thread_id", "n_questions"]:
            st.session_state.pop(k, None)
        st.rerun()
    st.divider()
    st.caption("무료 Groq 한도를 함께 쓰므로 질문 수를 제한합니다. 한도가 차면 잠시 뒤 다시 시도하세요.")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("thread_id", uuid.uuid4().hex)
st.session_state.setdefault("n_questions", 0)


def show_quota() -> None:
    left = max(MAX_QUESTIONS - st.session_state["n_questions"], 0)
    quota_slot.write(f"남은 질문: **{left}** / {MAX_QUESTIONS}")


show_quota()

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            st.caption(m["meta"])

if not st.session_state["messages"]:
    st.write("예시 질문")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state["pending"] = ex
            st.rerun()

question = st.chat_input("연말정산에 대해 물어보세요") or st.session_state.pop("pending", None)
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if st.session_state["n_questions"] >= MAX_QUESTIONS:
            reply, meta = "이 세션의 질문 한도에 도달했습니다. '대화 지우기'를 누르거나 나중에 다시 시도해 주세요.", ""
        else:
            try:
                agent, llm, log = get_agent(model_id)
                with st.spinner("문서를 찾고 답을 만드는 중..."):
                    ans = rc.ask(agent, llm, log, question, st.session_state["thread_id"])
                st.session_state["n_questions"] += 1
                reply = ans.text or "답변을 만들지 못했습니다. 질문을 다시 써 주세요."
                parts = []
                if ans.calls:
                    parts.append("사용한 도구: " + " / ".join(ans.calls))
                elif ans.tools:
                    parts.append("사용한 도구: " + ", ".join(dict.fromkeys(ans.tools)))
                if ans.pages:
                    parts.append("검색된 쪽: " + ", ".join(f"PDF {p}쪽" for p in ans.pages[:8]))
                if ans.blocked:
                    parts.append("가드레일이 차단했습니다")
                meta = " · ".join(parts)
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate" in msg.lower():
                    reply, meta = "지금 Groq 무료 한도가 차서 답할 수 없습니다. 잠시 뒤에 다시 시도해 주세요.", ""
                else:
                    reply, meta = f"오류가 났습니다 ({type(e).__name__}). 잠시 뒤에 다시 시도해 주세요.", ""
        st.markdown(reply)
        if meta:
            st.caption(meta)
    st.session_state["messages"].append({"role": "assistant", "content": reply, "meta": meta})
    show_quota()