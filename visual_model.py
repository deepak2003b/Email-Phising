import cv2, numpy as np
from sklearn.metrics.pairwise import cosine_similarity

phish_ref = np.load("trained/phishing_visual.npy")
legit_ref = np.load("trained/legitimate_visual.npy")

def visual_score(img_path):
    img = cv2.imread(img_path, 0)
    edges = cv2.Canny(img, 50, 150)
    feat = np.array([[np.sum(edges > 0) / edges.size,
                      len(cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)[0])]])
    p = cosine_similarity(feat, phish_ref).mean() 
    l = cosine_similarity(feat, legit_ref).mean()

    return float(p), float(l)