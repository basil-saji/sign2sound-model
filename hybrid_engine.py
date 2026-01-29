import numpy as np
import tensorflow as tf
from collections import deque
from enum import Enum

class SystemState(Enum):
    IDLE = 0
    STATIC_LOCK = 1
    DYNAMIC_PROCESS = 2

class HybridEngine:
    def __init__(
        self,
        static_path="alphabet_pose_mlp_24letters.keras",
        static_labels_path="alphabet_labels_24letters.npy",
        dynamic_path="mstcn_dynamic.keras",
        dynamic_labels_path="dynamic_labels.npy"
    ):
        # Load Models (flat directory)
        self.static_model = tf.keras.models.load_model(static_path)
        self.static_labels = np.load(static_labels_path, allow_pickle=True)

        self.dynamic_model = tf.keras.models.load_model(dynamic_path)
        self.dynamic_labels = np.load(dynamic_labels_path, allow_pickle=True)

        # Configuration
        self.STATIC_THRESHOLD = 0.90
        self.DYNAMIC_THRESHOLD = 0.85
        self.MOTION_THRESHOLD = 0.05
        self.BUFFER_SIZE = 30
        self.LOCKED_DIMENSION = 84

        # Runtime state
        self.buffer = deque(maxlen=self.BUFFER_SIZE)
        self.prev_features = None
        self.cooldown = 0

    def process_frame(self, feature_vector):
        if feature_vector.shape[0] != self.LOCKED_DIMENSION:
            print("Feature dimension mismatch")
            return None

        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        # Motion energy
        motion_energy = 0.0
        if self.prev_features is not None:
            motion_energy = np.linalg.norm(feature_vector - self.prev_features)

        self.prev_features = feature_vector.copy()

        # Static inference
        feat_input = feature_vector.reshape(1, -1)
        static_probs = self.static_model.predict(feat_input, verbose=0)[0]
        static_conf = np.max(static_probs)
        static_class = self.static_labels[np.argmax(static_probs)]

        if static_conf > self.STATIC_THRESHOLD and motion_energy < self.MOTION_THRESHOLD:
            self.buffer.clear()
            return {
                "type": "STATIC",
                "class": static_class,
                "confidence": float(static_conf)
            }

        # Dynamic inference
        self.buffer.append(feature_vector)

        if len(self.buffer) == self.BUFFER_SIZE:
            seq_input = np.array(self.buffer).reshape(1, self.BUFFER_SIZE, -1)
            dyn_probs = self.dynamic_model.predict(seq_input, verbose=0)[0]
            dyn_conf = np.max(dyn_probs)
            dyn_class = self.dynamic_labels[np.argmax(dyn_probs)]

            if dyn_conf > self.DYNAMIC_THRESHOLD:
                self.buffer.clear()
                self.cooldown = 15
                return {
                    "type": "DYNAMIC",
                    "class": dyn_class,
                    "confidence": float(dyn_conf)
                }

        return None
