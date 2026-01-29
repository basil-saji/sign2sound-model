import threading
import time
import math

class AsyncWordPredictor(threading.Thread):
    def __init__(self, vocab_memory):
        super().__init__()
        self.vocab = vocab_memory
        self.daemon = True
        
        self.current_prefix = ""
        self.current_history = []
        self.latest_suggestions = []
        
        self.running = True
        self.lock = threading.Lock()
        self.needs_update = False

    def update_input(self, prefix, history):
        with self.lock:
            if prefix != self.current_prefix or history != self.current_history:
                self.current_prefix = prefix.upper()
                self.current_history = list(history)
                self.needs_update = True

    def get_suggestions(self):
        with self.lock:
            return list(self.latest_suggestions)

    def run(self):
        while self.running:
            if self.needs_update:
                with self.lock:
                    prefix = self.current_prefix
                    history = self.current_history
                    self.needs_update = False 
                
                results = self._compute_smart_predictions(prefix, history)
                
                with self.lock:
                    self.latest_suggestions = results
            
            time.sleep(0.01)

    def _compute_smart_predictions(self, prefix, history, top_k=3):
        candidates = {}
        now = time.time()

        # Helper to safely check history
        def get_hist(idx):
            if len(history) >= abs(idx):
                return history[idx]
            return None

        def add_score(word, base_points):
            if prefix and not word.startswith(prefix): return
            if word not in candidates: candidates[word] = 0.0
            
            # --- FEATURE 1: RECENCY BIAS ---
            # Recent words (last 5 mins) get huge bonus
            recency_bonus = 0.0
            if word in self.vocab.data["user_words"]:
                last_used = self.vocab.data["user_words"][word].get("last_used", 0)
                if (now - last_used) < 300: 
                    recency_bonus = 30.0
            
            # --- FEATURE 2: REPETITION PENALTY ---
            # Stop loops like "IS IS IS"
            if get_hist(-1) == word:
                base_points -= 500.0 
            if get_hist(-2) == word:
                base_points -= 50.0

            candidates[word] += base_points + recency_bonus

        # --- FEATURE 3: DEEP CONTEXT ---
        context_found = False
        max_n = self.vocab.n 
        
        for i in range(max_n - 1, 0, -1):
            if len(history) < i: continue
                
            context_str = " ".join(history[-i:])
            
            if context_str in self.vocab.data["ngrams"]:
                suggestions = self.vocab.data["ngrams"][context_str]
                
                # Multiplier: Longer context = Higher score
                tier_score = 100.0 * i 
                
                for next_word, count in suggestions.items():
                    freq_boost = math.log(count + 1) * 10.0
                    add_score(next_word, tier_score + freq_boost)
                    context_found = True

        # --- FEATURE 4: FALLBACK ---
        should_show_generics = (prefix != "") or (not context_found)

        if should_show_generics:
            all_words = list(self.vocab.data["user_words"].keys()) + \
                        list(self.vocab.data["core_words"].keys())
            
            for word in all_words:
                if word in candidates and candidates[word] > 50: continue
                if prefix and not word.startswith(prefix): continue

                freq = 0
                if word in self.vocab.data["user_words"]:
                    freq = self.vocab.data["user_words"][word]["frequency"]
                
                score = math.log(freq + 1) * 2.0
                add_score(word, score)

        # --- FEATURE 5: EXACT MATCH ---
        if prefix and prefix in candidates:
             candidates[prefix] += 20.0 

        sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_cands[:top_k]]

    def stop(self):
        self.running = False