import cv2, os, numpy as np

def extract_visual_features(img_path):
    img = cv2.imread(img_path, 0)
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [edge_density, len(contours)]

os.makedirs("trained", exist_ok=True)

for label in ["phishing", "legitimate"]:
    features = []
    folder = f"dataset/{label}"
    for f in os.listdir(folder):
        features.append(extract_visual_features(os.path.join(folder, f)))
    np.save(f"trained/{label}_visual.npy", np.array(features))

print("✅ Visual model trained")