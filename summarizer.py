# app/src/main/python/summarizer.py
import numpy as np
import onnxruntime as ort
from preprocess import Preprocessing

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
        # Tokenisasi
        tokenized = self.preprocessing.tokenize_single_sentence(sentence)
        
        # Siapkan input untuk model
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        token_type_ids = tokenized['token_type_ids']
        
        # Jalankan model
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        
        # Cek apakah model punya multiple inputs
        if len(self.model.get_inputs()) == 3:
            # Model dengan 3 input (input_ids, attention_mask, token_type_ids)
            outputs = self.model.run(
                [output_name],
                {
                    self.model.get_inputs()[0].name: input_ids,
                    self.model.get_inputs()[1].name: attention_mask,
                    self.model.get_inputs()[2].name: token_type_ids
                }
            )
        else:
            # Model dengan 1 input
            outputs = self.model.run([output_name], {input_name: input_ids})
        
        logits = outputs[0]
        
        # Apply softmax untuk dapat probabilitas
        probs = self.softmax(logits)
        
        # Ambil probabilitas kelas positif (indeks 1)
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
        
        # Urutkan berdasarkan skor tertinggi
        ranked = sorted(ranked_sentences, key=lambda x: x["score"], reverse=True)
        
        selected = []
        current_chars = 0
        
        for item in ranked:
            selected.append(item)
            current_chars += len(item["sentence"])
            if current_chars >= target_chars:
                break
        
        # Urutkan kembali berdasarkan posisi awal
        selected = sorted(selected, key=lambda x: x["idx"])
        
        return selected
    
    def generate_summary(self, text):
        """Generate ringkasan dari teks"""
        # Bersihkan teks
        clean_text = self.preprocessing.text_cleaning(text)
        
        # Pecah jadi kalimat
        sentences = self.preprocessing.split_sentences(clean_text)
        
        if not sentences:
            return "Teks tidak valid atau terlalu pendek"
        
        # Beri skor
        ranked = self.rank_sentences(sentences)
        
        # Pilih kalimat terbaik
        selected = self.select_sentences(ranked)
        
        # Gabung jadi ringkasan
        summary = " ".join(item["sentence"] for item in selected)
        
        return summary

# ===== GLOBAL FUNCTIONS untuk dipanggil dari Kotlin =====

_model_session = None
_preprocessing = None
_summarizer = None

def get_summarizer():
    global _model_session, _preprocessing, _summarizer
    
    if _summarizer is None:
        # Load model ONNX
        _model_session = ort.InferenceSession('model.onnx')
        
        # Inisialisasi preprocessing
        _preprocessing = Preprocessing()
        
        # Buat summarizer
        _summarizer = SummaryGenerator(_model_session, _preprocessing, compression_ratio=0.27)
    
    return _summarizer

def ringkas_teks(teks_input):
    """
    Fungsi utama yang dipanggil dari Kotlin
    teks_input: string (bisa dari PDF atau input manual)
    """
    summarizer = get_summarizer()
    hasil = summarizer.generate_summary(teks_input)
    return hasil