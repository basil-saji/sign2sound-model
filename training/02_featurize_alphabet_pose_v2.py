import numpy as np
import os
import glob
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
DATA_DIR = "/content/alphabet_landmarks_static"
DRIVE_MODEL_DIR = "/content/drive/MyDrive/Sign2Sound_Hybrid/models/alphabet/pose_mlp_v2"
os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)

LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # 24 classes

# ================= FEATURE ENGINEERING =================
# This matches the "Alphabet Pose Model v2" plan
TIP_IDXS = [4, 8, 12, 16, 20]

def angle(a, b, c, eps=1e-8):
    """Calculate angle at point b given points a, b, c"""
    ba = a - b
    bc = c - b
    ba_n = ba / (np.linalg.norm(ba) + eps)
    bc_n = bc / (np.linalg.norm(bc) + eps)
    cosang = np.clip(np.dot(ba_n, bc_n), -1.0, 1.0)
    return np.arccos(cosang)

def featurize_pose(x63):
    """
    Convert 63 raw xyz coords -> 81 geometry features
    (Normalized coords + palm normal + tip distances + inter-tip dists + angles)
    """
    # Reshape to (21, 3)
    pts = x63.reshape(21, 3).copy()

    # 1. Wrist Normalization (origin at wrist)
    wrist = pts[0].copy()
    pts -= wrist

    # 2. Scale Normalization (by palm size)
    scale = np.linalg.norm(pts[9]) + 1e-6 # dist to middle_mcp
    pts /= scale

    # 3. Palm Normal (Orientation)
    v1 = pts[5]  # index_mcp
    v2 = pts[17] # pinky_mcp
    normal = np.cross(v1, v2)
    normal /= (np.linalg.norm(normal) + 1e-6)

    # 4. Fingertip Distances from Wrist
    tip_d = [np.linalg.norm(pts[i]) for i in TIP_IDXS]

    # 5. Inter-Fingertip Distances (Spread)
    tips = [pts[i] for i in [8, 12, 16, 20]]
    inter = []
    for i in range(len(tips)):
        for j in range(i+1, len(tips)):
            inter.append(np.linalg.norm(tips[i] - tips[j]))

    # 6. Joint Angles (MCP-PIP-DIP) for curling check
    angs = []
    for (a,b,c) in [(5,6,7), (9,10,11), (13,14,15), (17,18,19)]:
        angs.append(angle(pts[a], pts[b], pts[c]))

    # Flatten normalized coords
    coords = pts.reshape(-1)

    # Combine all features (63 + 3 + 5 + 6 + 4 = 81 dims)
    feat = np.concatenate([
        coords,
        normal,
        np.array(tip_d),
        np.array(inter),
        np.array(angs)
    ]).astype(np.float32)

    return feat

print("Feature extraction function ready.")
