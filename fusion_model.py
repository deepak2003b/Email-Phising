def fuse_scores(vp, vl, tp, tl):
    phishing_score = (vp + tp) / 2
    legit_score = (vl + tl) / 2
    confidence = abs(phishing_score - legit_score)
    result = "Phishing" if phishing_score > legit_score else "Legitimate"
    return result, phishing_score, legit_score, confidence