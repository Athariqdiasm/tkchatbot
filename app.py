# -*- coding: utf-8 -*-
"""
Bu Tanya — Asisten Informasi TK Kupu-Kupu Mungil
Chatbot customer service lembaga TK berbasis RAG (Gemini API + Streamlit).

Final Project — LLM-Based Tools and Gemini API Integration for Data Scientists (Hacktiv8)
"""

import os
import numpy as np
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types

# ----------------------------- Konfigurasi dasar -----------------------------
PDF_PATH = os.path.join("profil_kupu_kupu_mungil.pdf")
CHAT_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"
TOP_K = 4

st.set_page_config(page_title="Bu Tanya — TK Kupu-Kupu Mungil",
                   page_icon="🦋", layout="centered")


def get_client() -> genai.Client:
    """Ambil API key dari st.secrets (deploy) atau env var (lokal)."""
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if not api_key:
        st.error("GEMINI_API_KEY belum diatur. Tambahkan di .streamlit/secrets.toml "
                 "atau environment variable.")
        st.stop()
    return genai.Client(api_key=api_key)


# ----------------------------- RAG: index dokumen ----------------------------
def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Potong teks jadi chunk beririsan supaya konteks antar-paragraf tidak putus."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


@st.cache_resource(show_spinner="Menyiapkan knowledge base sekolah…")
def build_index(pdf_path: str):
    """Baca PDF -> chunk -> embed sekali saja (di-cache selama app hidup)."""
    reader = PdfReader(pdf_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = chunk_text(full_text)

    client = get_client()
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=chunks,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    vectors = np.array([e.values for e in result.embeddings], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return chunks, vectors


def retrieve(query: str, chunks: list[str], vectors: np.ndarray, k: int = TOP_K):
    """Cosine similarity sederhana — cukup untuk korpus <100 chunk."""
    client = get_client()
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    q = np.array(result.embeddings[0].values, dtype=np.float32)
    q /= np.linalg.norm(q)
    scores = vectors @ q
    top_idx = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_idx]


# ----------------------------- Prompting ------------------------------------
def build_system_prompt(tone: str, context: str) -> str:
    tone_rule = (
        "Gunakan bahasa Indonesia yang formal, sopan, dan profesional."
        if tone == "Formal"
        else "Gunakan bahasa Indonesia yang santai, hangat, dan ramah — sapa dengan "
             "'Ayah/Bunda', boleh pakai emoji secukupnya."
    )
    return f"""Kamu adalah "Bu Tanya", asisten informasi resmi TK Kupu-Kupu Mungil
(Kelompok Bermain Kreatif Kupu-Kupu Mungil, Tambun Selatan, Kabupaten Bekasi).
Tugasmu menjawab pertanyaan orang tua/wali seputar sekolah: profil, jenjang (TK,
Playgroup, Daycare), program, pendaftaran, biaya, jadwal, fasilitas, dan kontak.

ATURAN PENTING:
1. Jawab HANYA berdasarkan KONTEKS di bawah. Jangan mengarang informasi.
2. Jika jawabannya tidak ada di konteks, katakan dengan jujur bahwa kamu belum
   memiliki informasinya, lalu arahkan ke Admin via WhatsApp 0822-9703-9627
   (Miss Khansa) atau email kbk.kupumungil@gmail.com.
3. {tone_rule}
4. Jawab ringkas dan jelas. Gunakan poin-poin bila informasinya berjenjang.
5. Kamu hanya melayani topik seputar sekolah ini. Tolak dengan sopan pertanyaan
   di luar topik tersebut.

KONTEKS:
{context}"""


def generate_answer(client: genai.Client, history: list[dict],
                    system_prompt: str, temperature: float) -> str:
    contents = [
        types.Content(role=("user" if m["role"] == "user" else "model"),
                      parts=[types.Part.from_text(text=m["content"])])
        for m in history
    ]
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        ),
    )
    return response.text


# ----------------------------- UI -------------------------------------------
st.title("🦋 Bu Tanya")
st.caption("Asisten informasi TK Kupu-Kupu Mungil untuk Ayah & Bunda — "
           "tanyakan pendaftaran, biaya, program, jadwal, dan lainnya.")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    tone = st.radio("Gaya bahasa", ["Santai & Ramah", "Formal"], index=0)
    temperature = st.slider(
        "Temperature (kreativitas jawaban)", 0.0, 1.0, 0.3, 0.1,
        help="Rendah = konsisten & faktual (disarankan untuk CS). "
             "Tinggi = lebih variatif.")
    if st.button("🧹 Bersihkan percakapan", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown(
        "**Tentang aplikasi**\n\n"
        "Chatbot RAG: jawaban di-*ground* ke dokumen profil sekolah "
        "(PDF), diproses dengan Gemini API.\n\n"
        "_Final Project — Hacktiv8 × AI Opportunity Fund._")

# Memory percakapan
if "messages" not in st.session_state:
    st.session_state.messages = []

# Index RAG (cached)
chunks, vectors = build_index(PDF_PATH)
client = get_client()

# Quick questions
st.markdown("**Pertanyaan cepat:**")
cols = st.columns(4)
quick = [("💰 Biaya", "Berapa rincian biaya sekolahnya?"),
         ("📝 Cara daftar", "Bagaimana cara dan syarat pendaftaran murid baru?"),
         ("🕗 Jam sekolah", "Jam berapa kegiatan sekolah dan jam kantornya?"),
         ("📚 Program", "Apa saja program unggulan sekolah ini?")]
pending = None
for col, (label, q) in zip(cols, quick):
    if col.button(label, use_container_width=True):
        pending = q

# Render riwayat
for m in st.session_state.messages:
    avatar = "🦋" if m["role"] == "assistant" else "🧑"
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

# Input user (ketikan atau tombol cepat)
user_input = st.chat_input("Tulis pertanyaan Ayah/Bunda di sini…")
if pending and not user_input:
    user_input = pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🦋"):
        with st.spinner("Bu Tanya sedang mengetik…"):
            try:
                context = "\n\n---\n\n".join(
                    retrieve(user_input, chunks, vectors))
                system_prompt = build_system_prompt(
                    "Formal" if tone == "Formal" else "Santai", context)
                answer = generate_answer(
                    client, st.session_state.messages,
                    system_prompt, temperature)
            except Exception as e:
                answer = ("Maaf, sedang ada kendala teknis. Silakan coba lagi, "
                          "atau hubungi Admin di WhatsApp 0822-9703-9627. "
                          f"\n\n_Detail: {e}_")
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
