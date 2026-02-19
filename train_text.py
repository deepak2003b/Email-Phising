import cv2, os, numpy as np, pytesseract
from sentence_transformers import SentenceTransformer
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
os.makedirs("trained", exist_ok=True)

def extract_text(img_path):
    img = cv2.imread(img_path)
    return pytesseract.image_to_string(img)

for label in ["phishing", "legitimate"]:
    texts = []
    folder = f"dataset/{label}"
    for f in os.listdir(folder):
        txt = extract_text(os.path.join(folder, f))
        if txt.strip():
            texts.append(txt)
    embeddings = model.encode(texts)
    np.save(f"trained/{label}_text.npy", embeddings)

print("✅ Text model trained")