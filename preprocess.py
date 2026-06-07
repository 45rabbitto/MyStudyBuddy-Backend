from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Tambahkan folder current ke path
sys.path.append(os.path.dirname(__file__))

app = FastAPI()

# CORS untuk Android
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== LOAD MODEL (ONCE ON STARTUP) ==========
print("=" * 50)
print("Loading MobileBERT ONNX model...")
try:
    summarizer = get_summarizer()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
print("=" * 50)

# ========== API RINGKASAN (TERIMA TEKS, BUKAN FILE) ==========
class SummarizeRequest(BaseModel):
    text: str   # ← Ini teks hasil ekstraksi PDFBox dari Android

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    """
    MENERIMA TEKS DARI ANDROID (HASIL PDFBOX)
    BUKAN FILE PDF!
    """
    text = request.text
    
    # Validasi
    if not text or len(text.strip()) < 50:
        raise HTTPException(
            status_code=400, 
            detail="Teks terlalu pendek (minimal 50 karakter)"
        )
    
    try:
        # Panggil model ringkasan MobileBERT ANDA
        summary = ringkas_teks(text)
        
        return {
            "success": True,
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ========== API CHATBOT ==========
class ChatRequest(BaseModel):
    question: str
    context: str   # ← Teks materi dari Firestore (hasil ekstraksi PDFBox)

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Menerima pertanyaan + konteks materi dari Android
    """
    question = request.question
    context = request.context
    
    if len(context) > 2000:
        context = context[:2000]
    
    # TODO: Integrasi dengan DeepSeek API
    answer = f"""Berdasarkan materi yang Anda pelajari:

📖 Konteks: "{context[:300]}..."

💬 Pertanyaan: "{question}"

📝 Jawaban: [Integrasi dengan DeepSeek API akan ditambahkan]

💡 Saran: Tambahkan DEEPSEEK_API_KEY di file .env untuk chatbot yang lebih cerdas."""
    
    return {"answer": answer}

# ========== HEALTH CHECK ==========
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_type": "MobileBERT ONNX (Extractive Summarization)"
    }

@app.get("/")
async def root():
    return {
        "message": "My Study Buddy AI Server Running!",
        "note": "Server ini menerima TEKS (bukan file PDF), karena ekstraksi PDF dilakukan di Android menggunakan PDFBox",
        "endpoints": {
            "POST /summarize": "Meringkas teks (hasil PDFBox dari Android)",
            "POST /chat": "Chatbot dengan konteks materi",
            "GET /health": "Cek status server"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("🚀 My Study Buddy AI Server Running!")
    print("📡 URL: http://localhost:8000")
    print("✅ Health check: http://localhost:8000/health")
    print("📝 Note: Server menerima TEKS, bukan file PDF")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)