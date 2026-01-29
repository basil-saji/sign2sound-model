import numpy as np


class GestureController:
    def __init__(self, buffer_size=10, cooldown_frames=20):
        self.buffer_size = buffer_size
        self.cooldown_frames = cooldown_frames

        self.gesture_buffer = []
        self.cooldown_counter = 0

        self.valid_gestures = {
            "THUMB_UP",
            "THUMB_DOWN",
            "PINCH",
            "OPEN_PALM",
            "SMART_SELECT",
        }

    def update_and_check(self, landmarks):
        gesture = self._detect_frame_gesture(landmarks)

        # Always update buffer
        self.gesture_buffer.append(gesture)
        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer.pop(0)

        # Block triggering during cooldown, not sensing
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return None

        required = int(self.buffer_size * 0.7)

        for g in self.valid_gestures:
            if self.gesture_buffer.count(g) >= required:
                return self._trigger(g)

        return None

    def _trigger(self, gesture):
        self.cooldown_counter = self.cooldown_frames
        self.gesture_buffer.clear()
        return gesture

    def is_potential_gesture(self):
        count = sum(1 for g in self.gesture_buffer if g in self.valid_gestures)
        return count >= int(self.buffer_size * 0.4)

    def refine_prediction(self, lm_list, predicted_label, confidence):
        return predicted_label

    def _detect_frame_gesture(self, lm):
        if lm is None or len(lm) < 21:
            return None

        WRIST = 0
        THUMB_TIP = 4
        INDEX_MCP, INDEX_TIP = 5, 8
        MID_MCP, MID_TIP = 9, 12
        RING_MCP, RING_TIP = 13, 16
        PINKY_MCP, PINKY_TIP = 17, 20

        def dist(i, j):
            return np.linalg.norm(np.array(lm[i]) - np.array(lm[j]))

        palm = dist(WRIST, MID_MCP) + 1e-6

        # Smart select
        if (
            dist(THUMB_TIP, INDEX_MCP) > palm * 0.4
            and dist(INDEX_TIP, WRIST) > dist(INDEX_MCP, WRIST) * 1.5
            and dist(PINKY_TIP, WRIST) > dist(PINKY_MCP, WRIST) * 1.5
            and dist(MID_TIP, WRIST) < dist(MID_MCP, WRIST) * 1.2
            and dist(RING_TIP, WRIST) < dist(RING_MCP, WRIST) * 1.2
        ):
            return "SMART_SELECT"

        fingers_folded = True
        for tip, mcp in [
            (INDEX_TIP, INDEX_MCP),
            (MID_TIP, MID_MCP),
            (RING_TIP, RING_MCP),
            (PINKY_TIP, PINKY_MCP),
        ]:
            if dist(tip, WRIST) > dist(mcp, WRIST):
                fingers_folded = False
                break

        if fingers_folded:
            if dist(THUMB_TIP, INDEX_MCP) > palm * 0.6:
                if lm[THUMB_TIP][1] < lm[WRIST][1] - palm * 0.3:
                    return "THUMB_UP"
                if lm[THUMB_TIP][1] > lm[WRIST][1] + palm * 0.3:
                    return "THUMB_DOWN"

        if not fingers_folded:
            if dist(THUMB_TIP, INDEX_TIP) < palm * 0.15:
                if all(
                    dist(tip, WRIST) < dist(MID_MCP, WRIST) * 1.3
                    for tip in (MID_TIP, RING_TIP, PINKY_TIP)
                ):
                    return "PINCH"

            if all(
                dist(tip, WRIST) > palm * 0.85
                for tip in (
                    THUMB_TIP,
                    INDEX_TIP,
                    MID_TIP,
                    RING_TIP,
                    PINKY_TIP,
                )
            ):
                if dist(INDEX_TIP, PINKY_TIP) > palm * 0.9:
                    return "OPEN_PALM"

        return None
