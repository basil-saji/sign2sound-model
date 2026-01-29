import nltk
from typing import List, Tuple
from nltk.corpus import wordnet as wn

# --- OPTIMIZED LOADING ---
# Only download if missing. Prevents "freezing" on startup.
try:
    wn.ensure_loaded()
except LookupError:
    print("Downloading NLTK dictionary data (One-time setup)...")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download('averaged_perceptron_tagger')

class BroadcastSignInterpreter:
    def __init__(self):
        self.pronouns = {"I", "HE", "SHE", "WE", "THEY", "IT", "YOU"}
        # Map subject pronouns to object pronouns (e.g., "HE" -> "HIM")
        self.obj_pronoun_map = {"I": "ME", "HE": "HIM", "SHE": "HER", "WE": "US", "THEY": "THEM"}
        
        self.body_parts = {
            "CHEEK", "HEAD", "HAND", "FACE", "LEG", "ARM",
            "BACK", "EYE", "EYES", "LIP", "LIPS", "STOMACH"
        }

        self.question_words = {"WHAT", "WHY", "HOW", "WHEN", "WHERE", "WHO"}
        self.aux_verbs = {"IS", "ARE", "AM", "DO", "DID", "WILL", "CAN", "COULD"}

        self.autocorrect = {
            "COMMING": "COMING", "IM": "I AM", "DONT": "DO NOT",
            "WHATSUP": "WHAT IS UP", "WATS": "WHAT IS", "UR": "YOUR"
        }

        self.emergency_context = {"PAIN", "FIRE", "ACCIDENT", "HURT", "BLOOD", "HELP"}

    def interpret(self, words: List[str]) -> Tuple[str, float]:
        if not words:
            return "", 0.0
            
        # Keep original copy, work on normalized version
        original_words = words[:]
        words = self._normalize(words)

        # 1. EMERGENCY (High Priority - Ignores Grammar)
        if self._is_emergency(words):
            return "Emergency assistance required!", 1.0

        # 2. QUESTION
        if self._is_question(words):
            return self._build_dynamic_question(words), 0.9

        # 3. IMPERATIVE (Commands like "Close Door")
        if self._is_imperative(words):
            return self._build_imperative(words), 0.85

        # 4. STATEMENT (Subject -> Verb -> Object)
        s = self._build_statement(words)
        if s:
            return s, 0.8

        # 5. FALLBACK (Just speak the raw words if unsure)
        clean = " ".join(original_words).capitalize() + "."
        return clean, 0.5

    # --- BUILDERS ---
    def _build_imperative(self, words):
        # Case: "HIT ME HEAD" -> "Hit me on the head."
        if len(words) >= 3 and words[2] in self.body_parts:
             verb = words[0].capitalize()
             obj = words[1].lower()
             body = words[2].lower()
             return f"{verb} {obj} on the {body}."
        
        # Generic: "CLOSE DOOR" -> "Close the door."
        verb = words[0].capitalize()
        rest = []
        for w in words[1:]:
            if self._is_noun(w):
                rest.append("the " + w.lower())
            else:
                rest.append(w.lower())
        return f"{verb} {' '.join(rest)}."

    def _build_statement(self, words):
        subject = None
        verb = None
        obj = None
        remaining_words = words[:]

        # Step 1: Find Subject
        for i, w in enumerate(remaining_words):
            if w in self.pronouns:
                subject = w.capitalize()
                remaining_words.pop(i)
                break
        
        # Step 2: Find Verb
        if subject:
            for i, w in enumerate(remaining_words):
                if self._is_verb(w):
                    verb = w.lower()
                    # Fix conjugation: "She EAT" -> "She EATS"
                    if subject in ["He", "She", "It"] and not verb.endswith("s"):
                         verb += "s" 
                    remaining_words.pop(i)
                    break

        # Step 3: Find Object
        if subject and verb:
            if remaining_words:
                raw_obj = remaining_words[0]
                # Fix pronoun case: "I LOVE HE" -> "I LOVE HIM"
                if raw_obj in self.obj_pronoun_map:
                    obj = self.obj_pronoun_map[raw_obj].lower()
                else:
                    obj = raw_obj.lower()
                return f"{subject} {verb} {obj}."
            return f"{subject} {verb}."

        return None

    def _build_dynamic_question(self, words):
        # Specific fixes
        if words[0] == "WHAT" and "NAME" in words: return "What is your name?"
        if words == ["HOW", "YOU"]: return "How are you?"

        # Generic Assembly: "WHERE BATHROOM" -> "Where is the bathroom?"
        q_word = words[0].capitalize()
        rest = " ".join(w.lower() for w in words[1:])
        return f"{q_word} {rest}?"

    # --- HELPERS ---
    def _is_question(self, words):
        return words[0] in self.question_words or words[0] in self.aux_verbs

    def _is_imperative(self, words):
        return self._is_verb(words[0]) and not self._is_question(words)

    def _is_emergency(self, words):
        return any(w in self.emergency_context for w in words)

    def _normalize(self, words):
        out = []
        for w in words:
            w = w.upper()
            if w in self.autocorrect:
                out.extend(self.autocorrect[w].split())
            else:
                out.append(w)
        return out

    def _is_noun(self, w):
        return bool(wn.synsets(w.lower(), pos=wn.NOUN))

    def _is_verb(self, w):
        return bool(wn.synsets(w.lower(), pos=wn.VERB))