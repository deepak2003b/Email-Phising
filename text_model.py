import cv2, pytesseract, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

phish_ref = np.load("trained/phishing_text.npy")
legit_ref = np.load("trained/legitimate_text.npy")

def text_score(img_path):
    img = cv2.imread(img_path)
    text = pytesseract.image_to_string(img)
    if not text.strip():
        return 0.0, 0.0
    emb = model.encode([text])
    return ( 
        float(cosine_similarity(emb, phish_ref).mean()), 
        float(cosine_similarity(emb, legit_ref).mean())
    )