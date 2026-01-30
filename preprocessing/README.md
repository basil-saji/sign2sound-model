# Data Preprocessing and Feature Engineering Pipeline

This directory contains the core engineering scripts used to transform raw image data into the optimized, rotation-invariant geometric feature vectors that power the Sign2Sound model.

The logic implemented in these scripts corresponds to the interactive experiments documented in the notebooks directory.

## Dataset Source and Acquisition

The model was trained on the **IEEE DataPort American Sign Language Dataset**. We specifically utilized the static image subset to ensure high-precision recognition of fingerspelling alphabets.

* **Dataset Name:** American Sign Language Dataset
* **Source:** [IEEE DataPort](https://ieee-dataport.org/documents/american-sign-language-dataset)
* **Classes:** 24 Static Alphabets (A-Y, excluding dynamic gestures J and Z).

### Important Note on Data Access
This repository does not include a script to automatically download the dataset due to its size and the authentication requirements of IEEE DataPort.

during the development phase (executed in **Google Colab**), the dataset was downloaded manually and hosted on Google Drive. The scripts in this directory are configured to read from that mounted drive structure.

**Dataset Directory Structure Used:**
The training scripts utilized the raw images located in the following specific directories within the unzipped dataset:
* `Augmented data/Train data 1`
* `Augmented data/Train data 2`

If reproducing the training locally, ensure your data paths in `01_extract_alphabet_landmarks_static.py` match your local directory structure.

## Preprocessing Pipeline

The workflow is divided into three modular stages to ensure data integrity and reproducibility.

### Stage 1: Landmark Extraction
**Script:** `01_extract_alphabet_landmarks_static.py`

This script serves as the interface between raw pixels and skeletal data.
* **Input:** Raw images from the IEEE Dataset folders (specifically the Augmented data subfolders mentioned above).
* **Process:**
    1.  Iterates recursively through class directories.
    2.  Applies **MediaPipe Hands** to detect 21 skeletal landmarks (x, y, z) per image.
    3.  Filters out low-quality images where detection confidence is below 0.5.
* **Output:** A raw intermediate dataset of skeletal coordinates.

### Stage 2: Geometric Feature Engineering
**Script:** `02_featurize_alphabet_pose_v2.py`

This is the critical feature engineering step. Instead of feeding raw coordinates to the model (which makes the model sensitive to hand position and camera rotation), we compute **Relative Geometric Features**.

* **Input:** Raw skeletal coordinates from Stage 1.
* **Transformation Logic:**
    * **Joint Angles (15 features):** Calculates the vector dot product between finger phalanges to determine the curl of each finger, independent of hand orientation.
    * **Surface Normals (Rotation Invariance):** Computes the cross-product of the wrist-index and wrist-pinky vectors to determine the palm's facing direction.
    * **Euclidean Distances (Scale Invariance):** Measures the distance from the wrist to every fingertip, normalized by the scale of the hand (Wrist to Middle Finger MCP).
* **Output:** A 63-dimensional vector for every sample that represents the *shape* of the hand, not its position.

### Stage 3: Dataset Preparation
**Script:** `03_prepare_alphabet_pose_features.py`

Finalizes the data structure for the Neural Network.
* **Input:** 63-dimensional geometric vectors.
* **Process:**
    1.  **Label Encoding:** Maps class names (e.g., "A", "B") to numeric indices (0-23).
    2.  **Stratified Splitting:** Divides data into Training (80%), Validation (10%), and Testing (10%) sets to ensure balanced class distribution.
    3.  **Shuffling:** Randomizes the order to prevent training bias.
* **Output:** Compressed .npy (NumPy) files containing `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, and `y_test`.

## Usage

To reproduce the dataset creation process, run the scripts in numerical order from the root directory:

```bash
# Step 1: Extract Landmarks
python preprocessing/01_extract_alphabet_landmarks_static.py

# Step 2: Compute Geometric Features
python preprocessing/02_featurize_alphabet_pose_v2.py

# Step 3: Split and Save
python preprocessing/03_prepare_alphabet_pose_features.py
