import json
import time
import os

class VocabularyMemory:
    def __init__(self, path="vocab_memory.json", n_order=5):
        self.path = path
        self.n = n_order 
        self.data = {
            "core_words": {},
            "user_words": {},
            "ngrams": {},
            "stats": {"created": time.time()}
        }
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    loaded = json.load(f)
                    for key in self.data:
                        if key in loaded:
                            self.data[key] = loaded[key]
            except:
                self._initialize_defaults()
        else:
            self._initialize_defaults()
            self.save()

    def _initialize_defaults(self):
        self.data["core_words"] = {"HELLO": {}, "YES": {}, "NO": {}}
        self.save()

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        except:
            pass

    def register_word(self, word):
        if not word: return
        word = word.upper().strip()
        timestamp = time.time()

        if word not in self.data["user_words"]:
            self.data["user_words"][word] = {"frequency": 1, "last_used": timestamp}
        else:
            self.data["user_words"][word]["frequency"] += 1
            self.data["user_words"][word]["last_used"] = timestamp
        self.save()

    def register_sequence(self, history_list):
        if len(history_list) < 2: return
        target_word = history_list[-1]
        
        for i in range(1, self.n): 
            if len(history_list) < i + 1: break
            context = " ".join(history_list[-(i+1):-1])
            
            if context not in self.data["ngrams"]:
                self.data["ngrams"][context] = {}
            if target_word not in self.data["ngrams"][context]:
                self.data["ngrams"][context][target_word] = 0
            
            self.data["ngrams"][context][target_word] += 1
        self.save()