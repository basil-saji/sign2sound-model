import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import sys
import time
import threading


# Import modules
try:
    sys.path.append('src')
except:
    pass


from features_alphabet import featurize_pose
from vocab_memory import VocabularyMemory
from word_predictor import AsyncWordPredictor
from gesture_utils import GestureController
from tts_engine import TTSEngine
from broadcast import Broadcaster
from broadcast_window import launch_broadcast_window


# Paths
MODEL_PATH = "models/alphabet_pose_mlp_24letters.keras"
LABELS_PATH = "models/alphabet_labels_24letters.npy"


# --- CONFIGURATION ---
SUPABASE_URL = "https://wqqckkuycvthvizcwfgn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndxcWNra3V5Y3Z0aHZpemN3ZmduIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjYxNDcxMDYsImV4cCI6MjA4MTcyMzEwNn0.d2mfBuqKG8g4NSLb-EMCnzd-U-_mH35FwOxsbjbuGQ8"

#Per-letter confidence thresholds 
CONFIDENCE_THRESHOLDS = {
    'N': 0.75,
    'M': 0.72,
    'T': 0.72,
    'S': 0.70,
    'default': 0.60
}

MODE_SPELLING = "SPELLING"
MODE_GESTURE = "GESTURE"


# --- UI HELPER FUNCTIONS ---
def draw_ui_box(overlay, x, y, w, h, color=(0, 0, 0), alpha=0.5):
    sub_img = overlay[y:y+h, x:x+w]
    rect = np.full(sub_img.shape, color, dtype=np.uint8)
    cv2.addWeighted(sub_img, 1 - alpha, rect, alpha, 0, sub_img)
    overlay[y:y+h, x:x+w] = sub_img


def draw_text(img, text, x, y, font_scale=0.6, color=(255, 255, 255), thickness=1):
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def main():
    print("Initializing Sign2Sound System...")


    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        labels = np.load(LABELS_PATH, allow_pickle=True)
    except Exception as e:
        print(f"Model load error: {e}")
        return


    # Initialize Memory and Async Predictor
    vocab = VocabularyMemory()
    predictor_thread = AsyncWordPredictor(vocab)
    predictor_thread.start()


    gesture_ctrl = GestureController()
    tts = TTSEngine()


    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils


    # State variables
    letter_buffer = []
    current_word = ""
    sentence_history = []
    last_confirmed_word = None
    suggestions = []
    
    is_new_sentence_start = False


    raw_pred_buffer = deque(maxlen=5)
    feature_buffer = deque(maxlen=3)
    last_detected_letter = None
    live_confidence = 0.0
    letter_hold_frames = 0
    LETTER_HOLD_THRESHOLD = 15
    REPEAT_DELAY_FRAMES = 45


    spelling_cooldown_frames = 0
    SPELLING_COOLDOWN_DURATION = 45
    last_activity_time = time.time()
    PAUSE_THRESHOLD = 1.5
    system_mode = MODE_SPELLING


    broadcaster = Broadcaster(SUPABASE_URL, SUPABASE_KEY)


    # Broadcast Window Thread
    def wait_and_launch():
        print("Waiting for broadcast connection...")
        while True:
            if broadcaster.session_id != "OFFLINE":
                launch_broadcast_window(broadcaster.session_id)
                break
            time.sleep(1)
            
    threading.Thread(target=wait_and_launch, daemon=True).start()


    cap = cv2.VideoCapture(0)
    print("System Ready. Press 'q' to quit.")


    while True:
        ret, frame = cap.read()
        if not ret: break


        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)


        status_msg = ""
        status_color = (200, 200, 200)
        live_letter_display = "_"
        live_confidence = 0.0
        system_mode = MODE_SPELLING


        if spelling_cooldown_frames > 0: spelling_cooldown_frames -= 1


        # --- UPDATE PREDICTOR (Non-blocking) ---
        clean_history = [w for w in sentence_history if w != "<SPACE>"]
        predictor_thread.update_input(current_word, clean_history)


        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            lm_list = [[p.x, p.y, p.z] for p in lm.landmark]


            gesture = gesture_ctrl.update_and_check(lm_list)
            
            # --- FETCH LATEST PREDICTIONS (Instant) ---
            suggestions = predictor_thread.get_suggestions()


            if gesture:
                system_mode = MODE_GESTURE
                last_activity_time = time.time()
                spelling_cooldown_frames = SPELLING_COOLDOWN_DURATION


                word_to_add = None
                
                # --- 1. THUMB UP (Confirm RAW) ---
                if gesture == "THUMB_UP":
                    if is_new_sentence_start:
                        sentence_history.clear()
                        is_new_sentence_start = False


                    if current_word:
                        word_to_add = current_word
                        status_msg = f"RAW INPUT: {word_to_add}"
                    elif suggestions:
                        word_to_add = suggestions[0]
                        status_msg = f"CONFIRMED: {word_to_add}"


                # --- 2. SMART SELECT (Confirm PREDICTION) ---
                elif gesture == "SMART_SELECT":
                    if is_new_sentence_start:
                        sentence_history.clear()
                        is_new_sentence_start = False


                    if suggestions:
                        word_to_add = suggestions[0]
                        status_msg = f"AUTO-COMPLETE: {word_to_add}"
                    elif current_word:
                        word_to_add = current_word
                        status_msg = f"RAW INPUT: {word_to_add}"


                # --- COMMON ADD WORD LOGIC ---
                if word_to_add:
                    sentence_history.append(word_to_add)
                    
                    # LEARN: Register word AND sequence
                    vocab.register_word(word_to_add)
                    vocab.register_sequence([w for w in sentence_history if w != "<SPACE>"])
                    
                    sentence_history.append("<SPACE>")
                    tts.speak(word_to_add)
                    broadcaster.send({"type": "word", "text": word_to_add})
                    
                    last_confirmed_word = word_to_add
                    letter_buffer.clear()
                    current_word = ""
                    
                    # Force update predictor immediately
                    predictor_thread.update_input("", [w for w in sentence_history if w != "<SPACE>"])
                    status_color = (0, 255, 0)


                # --- 3. OPEN PALM (Finish Sentence) ---
                elif gesture == "OPEN_PALM":
                    if sentence_history and not is_new_sentence_start:
                        broadcaster.send({"type": "full_sentence"})
                        is_new_sentence_start = True
                        status_msg = "SENTENCE COMPLETED"
                        status_color = (0, 255, 255)


                # --- 4. THUMB DOWN (Clear/Undo) ---
                elif gesture == "THUMB_DOWN":
                    if current_word:
                        letter_buffer.clear()
                        current_word = ""
                        status_msg = "CLEARED WORD"
                    elif sentence_history:
                        if sentence_history[-1] == "<SPACE>": sentence_history.pop()
                        if sentence_history: sentence_history.pop()
                        status_msg = "UNDO LAST WORD"
                    status_color = (0, 0, 255)


                # --- 5. PINCH (Backspace) ---
                elif gesture == "PINCH":
                    if letter_buffer:
                        letter_buffer.pop()
                        current_word = "".join(letter_buffer)
                        status_msg = "BACKSPACE"
                        status_color = (255, 100, 100)
                    elif sentence_history and not is_new_sentence_start:
                        if sentence_history[-1] == "<SPACE>": sentence_history.pop()
                        if sentence_history:
                            word_to_edit = sentence_history.pop()
                            broadcaster.send({"type": "undo_last_word"})
                            letter_buffer = list(word_to_edit)
                            if letter_buffer: letter_buffer.pop()
                            current_word = "".join(letter_buffer)
                            last_confirmed_word = None
                            status_msg = "EDITING PREVIOUS..."
                            status_color = (255, 165, 0)


            # CMD LOCK 
            if (
                gesture_ctrl.is_potential_gesture()
                and current_word == ""
                and spelling_cooldown_frames == 0
            ):
                draw_text(frame, "CMD LOCK", w - 140, 30, 0.7, (0, 0, 255), 2)


            elif spelling_cooldown_frames == 0 and system_mode == MODE_SPELLING:
                x_raw = np.array([c for pt in lm_list for c in pt], dtype=np.float32)
                # Calculate features for THIS frame
                this_frame_features = featurize_pose(x_raw).reshape(1, -1)
                # Add to buffer
                feature_buffer.append(this_frame_features)
                # Average the features across the buffer (Smoothing)
                avg_features = np.mean(np.array(feature_buffer), axis=0)
                # Predict on the AVERAGED features
                preds = model.predict(avg_features, verbose=0)[0]
                idx = np.argmax(preds)
                conf = preds[idx]

                # Confidence-aware acceptance (REPLACES if conf > 0.60:)
                detected_raw = labels[idx]
                required_conf = CONFIDENCE_THRESHOLDS.get(detected_raw, CONFIDENCE_THRESHOLDS['default'])

                if conf < required_conf:
                    continue  # reject unstable prediction

               
                
                # Tie-Breaker
                detected_refined = gesture_ctrl.refine_prediction(lm_list, detected_raw, conf)
                
                # Convert refined string back to index for buffer consistency
                try:
                    idx_refined = np.where(labels == detected_refined)[0][0]
                except:
                    idx_refined = idx # Fallback if label lookup fails
                
                # Add the REFINED index to buffer
                raw_pred_buffer.append(idx_refined)
                
                if len(raw_pred_buffer) >= 3:
                    detected = labels[max(set(raw_pred_buffer), key=raw_pred_buffer.count)]
                    live_letter_display = detected
                    live_confidence = conf


                    if detected == last_detected_letter:
                        letter_hold_frames += 1
                    else:
                        last_detected_letter = detected
                        letter_hold_frames = 1
                        
                    # Dynamic Thresholding
                    base_threshold = LETTER_HOLD_THRESHOLD

                    # If model is super sure (>85%), let it pass faster
                    if live_confidence > 0.85:
                        base_threshold = 8  # Fast mode

                    threshold = base_threshold
                    if letter_buffer and detected == letter_buffer[-1]:
                        threshold = REPEAT_DELAY_FRAMES
                        
                    if letter_hold_frames == threshold:
                        letter_buffer.append(detected)
                        current_word = "".join(letter_buffer)
                        last_activity_time = time.time()
                        letter_hold_frames = 0


        # --- IMPROVED UI RENDERING ---
        
        # 1. Background Overlay
        draw_ui_box(frame, 0, 0, w, 60, (20, 20, 20), 0.7)
        draw_ui_box(frame, 0, h - 100, w, 100, (20, 20, 20), 0.7)
        draw_ui_box(frame, w - 120, 70, 110, 140, (30, 30, 30), 0.6)


        # 2. History Display
        sent_str = "".join([" " if t == "<SPACE>" else t for t in sentence_history[-8:]])
        draw_text(frame, "HISTORY:", 15, 25, 0.5, (180, 180, 180), 1)
        draw_text(frame, sent_str, 15, 50, 0.8, (255, 255, 255), 2)


        # 3. Live Letter Panel
        draw_text(frame, "LIVE", w - 90, 95, 0.5, (0, 255, 255), 1)
        draw_text(frame, live_letter_display, w - 95, 155, 2.0, (0, 255, 255), 3)
        if live_letter_display != "_":
            conf_str = f"{int(live_confidence * 100)}%"
            draw_text(frame, conf_str, w - 85, 180, 0.5, (200, 200, 200), 1)
            
            if letter_hold_frames > 0:
                threshold = REPEAT_DELAY_FRAMES if (letter_buffer and live_letter_display == letter_buffer[-1]) else LETTER_HOLD_THRESHOLD
                progress = min(letter_hold_frames / threshold, 1.0)
                bar_width = int(80 * progress)
                cv2.rectangle(frame, (w - 105, 195), (w - 25, 200), (60, 60, 60), -1)
                cv2.rectangle(frame, (w - 105, 195), (w - 105 + bar_width, 200), (0, 255, 0), -1)


        # 4. Suggestions & Labels (UPDATED UI)
        if current_word:
            # Label: Thumb Up -> Raw
            cv2.putText(frame, "THUMB UP (RAW):", (15, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.rectangle(frame, (15, h - 60), (15 + len(current_word)*18 + 10, h - 30), (50, 50, 50), -1)
            cv2.putText(frame, current_word, (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


            # Label: Rock On -> Smart
            if suggestions:
                top_pred = suggestions[0]
                start_x = 250
                
                cv2.putText(frame, "ROCK ON (AUTO):", (start_x, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                cv2.rectangle(frame, (start_x, h - 60), (start_x + len(top_pred)*18 + 10, h - 30), (0, 100, 0), -1)
                cv2.putText(frame, top_pred, (start_x + 5, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
                
                if len(suggestions) > 1:
                    rem_text = " ".join(suggestions[1:])
                    # Fixed the tuple argument error
                    cv2.putText(frame, rem_text, (start_x, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        # New Feature: Show next-word prediction even when typing nothing
        elif not current_word and suggestions:
             top_pred = suggestions[0]
             start_x = 250
             cv2.putText(frame, "NEXT WORD?", (start_x, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
             cv2.rectangle(frame, (start_x, h - 60), (start_x + len(top_pred)*18 + 10, h - 30), (0, 100, 0), -1)
             cv2.putText(frame, top_pred, (start_x + 5, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)


        # 5. Status Message
        if status_msg:
            text_size = cv2.getTextSize(status_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            center_x = (w - text_size[0]) // 2
            draw_text(frame, status_msg, center_x, h - 120, 0.8, status_color, 2)
        elif current_word and (time.time() - last_activity_time > PAUSE_THRESHOLD):
            draw_text(frame, "Confirm?", w // 2 - 40, h - 120, 0.6, (200, 200, 200), 1)


        cv2.imshow("Sign2Sound System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


    # Cleanup
    predictor_thread.stop()
    predictor_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    tts.stop()


if __name__ == "__main__":
    main()
