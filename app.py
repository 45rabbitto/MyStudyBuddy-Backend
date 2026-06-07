from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import re
from pathlib import Path
import sys
import os

# ========== PREPROCESSING CLASS ==========
class Preprocessing:
    def __init__(self):
        # Load MobileBERT vocab dari file
        self.vocab = self.load_vocab()
        self.max_length = 128

    def load_vocab(self):
        """Load vocab dari file txt"""
        vocab_path = Path(__file__).parent / "mobilebert_tokenizer" / "vocab.txt"
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f]
        return {word: idx for idx, word in enumerate(vocab)}

    def tokenize_single_sentence(self, text):
        """Tokenize satu kalimat tanpa library transformers"""
        # Lowercase untuk MobileBERT uncased
        text = text.lower()

        # Tokenisasi sederhana (split kata dan tanda baca)
        tokens = []
        for match in re.finditer(r'\w+|[^\w\s]', text):
            token = match.group()
            tokens.append(token)

        # Convert ke token IDs
        input_ids = []
        for token in tokens:
            if token in self.vocab:
                input_ids.append(self.vocab[token])
            else:
                input_ids.append(self.vocab.get('[UNK]', 100))

        # Format BERT: [CLS] + tokens + [SEP]
        cls_id = self.vocab.get('[CLS]', 101)
        sep_id = self.vocab.get('[SEP]', 102)
        pad_id = self.vocab.get('[PAD]', 0)

        input_ids = [cls_id] + input_ids + [sep_id]

        # Padding atau truncation
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            input_ids[-1] = sep_id
        else:
            input_ids = input_ids + [pad_id] * (self.max_length - len(input_ids))

        # Attention mask
        attention_mask = [1] * min(len(input_ids), self.max_length)
        attention_mask = attention_mask + [0] * (self.max_length - len(attention_mask))

        # Token type ids
        token_type_ids = [0] * self.max_length

        return {
            'input_ids': np.array([input_ids], dtype=np.int64),
            'attention_mask': np.array([attention_mask], dtype=np.int64),
            'token_type_ids': np.array([token_type_ids], dtype=np.int64)
        }

    def split_sentences(self, text):
        """Memecah teks menjadi list kalimat"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def text_cleaning(self, text):
        """Bersihkan teks"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text


# ========== SUMMARY GENERATOR CLASS ==========
class SummaryGenerator:
    def __init__(self, model_session, preprocessing, compression_ratio=0.27):
        self.model = model_session
        self.preprocessing = preprocessing
        self.compression_ratio = compression_ratio

    def softmax(self, logits):
        """Softmax function untuk convert logits ke probabilitas"""
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum(axis=-1, keepdims=True)

    def predict_score(self, sentence):
        """Prediksi skor pentingnya sebuah kalimat (0-1)"""
        tokenized = self.preprocessing.tokenize_single_sentence(sentence)

        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        token_type_ids = tokenized['token_type_ids']

        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name

        if len(self.model.get_inputs()) == 3:
            outputs = self.model.run(
                [output_name],
                {
                    self.model.get_inputs()[0].name: input_ids,
                    self.model.get_inputs()[1].name: attention_mask,
                    self.model.get_inputs()[2].name: token_type_ids
                }
            )
        else:
            outputs = self.model.run([output_name], {input_name: input_ids})

        logits = outputs[0]
        probs = self.softmax(logits)

        return float(probs[0][1])

    def rank_sentences(self, sentences):
        """Beri skor untuk setiap kalimat"""
        ranked = []
        for idx, sentence in enumerate(sentences):
            score = self.predict_score(sentence)
            ranked.append({
                "idx": idx,
                "sentence": sentence,
                "score": score
            })
        return ranked

    def select_sentences(self, ranked_sentences):
        """Pilih kalimat berdasarkan skor hingga mencapai compression_ratio"""
        total_chars = sum(len(item["sentence"]) for item in ranked_sentences)
        target_chars = int(total_chars * self.compression_ratio)

        ranked = sorted(ranked_sentences, key=lambda x: x["score"], reverse=True)

        selected = []
        current_chars = 0

        for item in ranked:
            selected.append(item)
            current_chars += len(item["sentence"])
            if current_chars >= target_chars:
                break

        selected = sorted(selected, key=lambda x: x["idx"])

        return selected

    def generate_summary(self, text):
        """Generate ringkasan dari teks"""
        clean_text = self.preprocessing.text_cleaning(text)
        sentences = self.preprocessing.split_sentences(clean_text)

        if not sentences:
            return "Teks tidak valid atau terlalu pendek"

        ranked = self.rank_sentences(sentences)
        selected = self.select_sentences(ranked)
        summary = " ".join(item["sentence"] for item in selected)

        return summary


# ========== GLOBAL VARIABLES ==========
_model_session = None
_preprocessing = None
_summarizer = None

def get_summarizer():
    global _model_session, _preprocessing, _summarizer

    if _summarizer is None:
        # Load model ONNX
        model_path = Path(__file__).parent / "model.onnx"
        _model_session = ort.InferenceSession(str(model_path))
        _preprocessing = Preprocessing()
        _summarizer = SummaryGenerator(_model_session, _preprocessing, compression_ratio=0.27)

    return _summarizer

def ringkas_teks(teks_input):
    summarizer = get_summarizer()
    hasil = summarizer.generate_summary(teks_input)
    return hasil


# ========== FASTAPI APP ==========
app = FastAPI()

# CORS untuk Android
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== LOAD MODEL ON STARTUP ==========
print("=" * 50)
print("Loading MobileBERT ONNX model...")
try:
    summarizer = get_summarizer()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
print("=" * 50)

# ========== API ENDPOINTS ==========
class SummarizeRequest(BaseModel):
    text: str

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    """
    Meringkas teks menggunakan MODEL LOKAL MobileBERT ONNX
    """
    text = request.text

    if not text or len(text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Teks terlalu pendek (minimal 50 karakter)"
        )

    try:
        summary = ringkas_teks(text)

        return {
            "success": True,
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "model": "MobileBERT ONNX (Local)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

class ChatRequest(BaseModel):
    question: str
    context: str

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chatbot - untuk sementara respons sederhana
    """
    question = request.question
    context = request.context

    if len(context) > 2000:
        context = context[:2000]

    answer = f"""Berdasarkan materi yang Anda pelajari:

Dari materi: "{context[:300]}..."

Pertanyaan: "{question}"


Untuk chatbot yang lebih baik, integrasikan dengan DeepSeek API."""
    return {"answer": answer}

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": _summarizer is not None,
        "model_type": "MobileBERT ONNX (Local - Extractive Summarization)"
    }

@app.get("/")
async def root():
    return {
        "message": "My Study Buddy AI Server Running!",
        "summary_model": "MobileBERT ONNX (LOCAL)",
        "endpoints": {
            "POST /summarize": "Ringkasan teks",
            "POST /chat": "Chatbot",
            "GET /health": "Cek status"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = 8000
    print("\n" + "=" * 50)
    print("🚀 My Study Buddy AI Server Running!")
    print(f"📡 URL: http://localhost:{port}")
    print(f"✅ Health check: http://localhost:{port}/health")
    print("📝 Model Ringkasan: MobileBERT ONNX (LOKAL)")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)