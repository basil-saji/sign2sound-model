import mediapipe as mp
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

# ================= CONFIGURATION =================
INPUT_ROOT = "/content/static_dataset/Augmented Data"
OUTPUT_DIR = "/content/drive/MyDrive/Sign2Sound_Hybrid/datasets/alphabet_landmarks_static"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mediapipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # Essential for static images
    max_num_hands=1,
    min_detection_confidence=0.5
)

# Target Letters (A-Y, excluding J, Z)
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# Specific subfolders to use (Fair & Dark skin datasets)
TARGET_SUBFOLDERS = [
    "Train Data 1",  # Fair skin
    "Train Data 2"   # Dark skin
]

# ================= EXTRACTION FUNCTION =================
def extract_landmarks():
    print(f"Searching in: {INPUT_ROOT}")
    print(f"Target subfolders: {TARGET_SUBFOLDERS}")

    for letter in LETTERS:
        print(f"\nProcessing Letter: {letter}")
        all_landmarks_for_letter = []

        # Iterate through both Train Data 1 and Train Data 2
        for subfolder in TARGET_SUBFOLDERS:
            folder_path = os.path.join(INPUT_ROOT, subfolder, letter)

            if not os.path.exists(folder_path):
                print(f"Folder not found: {subfolder}/{letter}")
                continue

            # Get images
            images = glob.glob(os.path.join(folder_path, "*"))
            valid_images = [f for f in images if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            print(f"   found {len(valid_images)} images in {subfolder}...")

            # Process images
            for img_path in tqdm(valid_images, desc=f"   Extracting {subfolder}/{letter}", leave=False):
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        continue

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image_rgb)

                    if results.multi_hand_landmarks:
                        lm_list = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            lm_list.extend([lm.x, lm.y, lm.z])

                        all_landmarks_for_letter.append(lm_list)

                except Exception as e:
                    continue

        # Save accumulated landmarks for this letter
        if all_landmarks_for_letter:
            npy_path = os.path.join(OUTPUT_DIR, f"{letter}.npy")
            np_data = np.array(all_landmarks_for_letter, dtype=np.float32)
            np.save(npy_path, np_data)
            print(f"Saved {letter}.npy: {np_data.shape} samples")
        else:
            print(f"No landmarks found for {letter} in any folder")
# ================= RUN =================
extract_landmarks()
hands.close()
print("\nExtraction Done")
