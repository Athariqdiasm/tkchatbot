# 🦋 Bu Tanya — Chatbot Customer Service TK Kupu-Kupu Mungil

Chatbot layanan informasi untuk orang tua/wali murid **TK Kupu-Kupu Mungil**
(Kelompok Bermain Kreatif Kupu-Kupu Mungil, Tambun Selatan, Kab. Bekasi),
dibangun dengan **Streamlit + Gemini API + RAG (Retrieval-Augmented Generation)**.

> Final Project — *LLM-Based Tools and Gemini API Integration for Data Scientists*
> (Hacktiv8 × AI Opportunity Fund: Asia Pacific).

## ✨ Fitur

| Fitur | Implementasi |
|---|---|
| Domain knowledge spesifik | RAG dari dokumen profil sekolah (`data/profil_kupu_kupu_mungil.pdf`) |
| Jawaban *grounded* (anti-halusinasi) | Model hanya menjawab dari konteks retrieval; di luar itu diarahkan ke admin |
| Gaya bahasa | Toggle **Santai & Ramah** ↔ **Formal** (sidebar) |
| Konfigurasi parameter | Slider **temperature** di sidebar |
| Memory percakapan | `st.session_state` — mendukung pertanyaan lanjutan |
| UX layanan CS | Quick-question buttons (Biaya, Cara Daftar, Jam Sekolah, Program) |

## 🏗️ Arsitektur

```
Orang tua → Streamlit chat UI (+ quick buttons, sidebar params)
              ↓
        st.session_state (memory percakapan)
              ↓
   PDF profil sekolah → chunking (800 char, overlap 150)
              ↓
   Gemini Embedding (gemini-embedding-001) → vektor (cached)
              ↓
   Cosine similarity → top-4 chunk relevan
              ↓
   Gemini (gemini-2.5-flash) + system prompt persona "Bu Tanya"
              ↓
   Jawaban grounded → UI
```

### Design decisions (mengapa begini)

- **Retrieval pakai cosine similarity NumPy, bukan vector DB eksternal.**
  Korpus hanya 1 PDF (±20 chunk). FAISS/Chroma/Pinecone di skala ini menambah
  dependency tanpa manfaat — konsepnya identik (embedding → nearest neighbor),
  footprint-nya jauh lebih kecil, dan deploy ke Streamlit Cloud lebih andal.
- **Grounded-only answering.** Chatbot CS lembaga yang mengarang biaya adalah
  liability. Jika informasi tidak ada di knowledge base, bot secara eksplisit
  mengaku tidak tahu dan mengarahkan ke WhatsApp admin.
- **Temperature default 0.3.** Use case CS menuntut jawaban konsisten dan
  faktual; slider tetap disediakan sebagai parameter yang bisa dieksplorasi.

## 🚀 Cara Menjalankan (Lokal)

```bash
git clone <URL-REPO-INI>
cd tk-chatbot
pip install -r requirements.txt

# Set API key (dapatkan dari https://aistudio.google.com/apikey)
mkdir -p .streamlit
echo 'GEMINI_API_KEY = "ISI_API_KEY_ANDA"' > .streamlit/secrets.toml

streamlit run app.py
```

> ⚠️ Jangan commit `secrets.toml` ke repo — sudah masuk `.gitignore`.

## ☁️ Deploy ke Streamlit Community Cloud

1. Push repo ini ke GitHub (public).
2. Buka [share.streamlit.io](https://share.streamlit.io) → **New app** → pilih repo → main file `app.py`.
3. Di **Advanced settings → Secrets**, isi:
   ```toml
   GEMINI_API_KEY = "ISI_API_KEY_ANDA"
   ```
4. Deploy — selesai.

## 🖼️ Screenshots

| Tampilan | Gambar |
|---|---|
| Halaman utama & quick buttons | ![home](screenshots/01_home.png) |
| Percakapan (mode santai) | ![chat](screenshots/02_chat.png) |
| Jawaban di luar knowledge base | ![fallback](screenshots/03_fallback.png) |

## 📄 Knowledge Base

`data/profil_kupu_kupu_mungil.pdf` (4 halaman) disusun dari informasi publik
situs lembaga (kupukupumungil.wordpress.com & kupukupumungil.sch.id): profil,
visi-misi CITA, jenjang TK/PG/Daycare, program unggulan (Tahfidz Juz 28–30,
Camp Qur'an, TIK, Market Day), pendaftaran, FAQ, dan kontak.
**Bagian biaya & sebagian jadwal bersifat ilustratif (dummy)** untuk keperluan
demo — bukan tarif resmi lembaga.

## 🧰 Tech Stack

`Python` · `Streamlit` · `google-genai (Gemini API)` · `pypdf` · `NumPy`
