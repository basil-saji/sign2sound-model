# Sign2Sound: Precision-First Sign Language Communication System

Sign2Sound is a specialized communication tool designed to bridge the gap between static sign language (fingerspelling) and fluent digital conversation. Unlike standard recognition models that output raw, jittery characters, our system integrates **Geometric Feature Extraction**, **Predictive Text Intelligence**, and **Real-Time Broadcasting** to create a usable product for accessibility.

## 🚀 Key Features

### 1. Zero-Latency Fingerspelling
* **Geometric Feature Extraction:** Instead of raw coordinates, we process relative angles and distances between hand landmarks. This makes the system robust to camera rotation and varying hand sizes.
* **Precision-First Inference:** Optimized for CPU performance, running locally on standard laptops without GPU requirements.

### 2. Intelligent Context Prediction
Typing letter-by-letter is slow. We mitigate this with a custom **N-Gram Predictive Engine**:
* **Smart Autocomplete:** Suggests full words based on the first few characters.
* **Context Awareness:** Predicts the *next* word in a sentence based on previous input (e.g., "WE" -> suggests "ARE").
* **Adaptive Memory:** The system learns from user usage over time, prioritizing frequently used vocabulary.

### 3. Remote Broadcasting (The "Studio" Feature)
We solved the problem of "who sees the text?" by decoupling the display from the camera:
* **Live Web Sync:** The Python client pushes recognized text to a cloud database (Supabase) in real-time.
* **PWA Receiver:** Any device (phone, tablet, external monitor) can scan a QR code to view the live transcript instantly.
* **Accessibility First:** Allows a deaf signer to communicate with an audience who can read the output on their own devices.

---

## 📂 Repository Structure

| Directory | Description |
| :--- | :--- |
| `src/` | Core logic modules (Feature extraction, Prediction, Broadcasting). |
| `models/` | The trained `.keras` Geometric MLP and label encoder. |
| `training/` | Training scripts and performance metrics. |
| `main.py` | The main inference application entry point. |

---

## 🛠️ Installation & Usage

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

```bash
# Clone the repository
git clone [https://github.com/basil-saji/sign2sound-model.git](https://github.com/basil-saji/sign2sound-model.git)
cd sign2sound-model

# Install dependencies
pip install -r requirements.txt
