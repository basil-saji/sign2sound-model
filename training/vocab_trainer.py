import json
import time
import os

# --- CONFIGURATION ---
TARGET_FILE = "vocab_memory.json"
N_ORDER = 5  # Learn sequences up to 5 words long


YOUR_CUSTOM_DATA = [
"That is a valid point",
"I see what you mean",
"Let us discuss this",
"I have a suggestion",
"That sounds reasonable",
"Please clarify your point",
"I understand your perspective",
"We should consider that",
"That is a great idea",
"I disagree with that",
"We need a better plan",
"Let us start the meeting",
"Is this the final version",
"Please send the document",
"I will check the details",
"That is very helpful",
"We can move forward now",
"What is the next step",
"Let us review the goals",
"I appreciate your input",
"This is a top priority",
"We should be very careful",
"That makes perfect sense",
"I am not quite sure",
"Let us look at facts",
"Please explain the process",
"We need more information",
"That is an excellent observation",
"I agree with your assessment",
"Let us stay on track",
"This requires more thought",
"I have a brief question",
"Please update the status",
"We should finalize this today",
"That is a fair critique",
"I will handle that task",
"Let us coordinate our efforts",
"This is the main objective",
"I value your professional opinion",
"We need a clear solution",
"That is a significant improvement",
"Please provide some feedback",
"I will follow up later",
"Let us simplify the approach",
"This seems very promising",
"We should wait for results",
"That is a logical conclusion",
"I see a potential issue",
"Let us address this now",
"Please share your thoughts",
"We are making great progress",
"That is a solid plan",
"I am ready to begin",
"Let us confirm the schedule",
"This is the correct way",
"We need to focus more",
"That is a tough challenge",
"I will manage the project",
"Let us define the scope",
"This is very important work",
"Please verify the data",
"We should seek more advice",
"That is a wise choice",
"I am in full agreement",
"Let us explore other options",
"This is the best strategy",
"We need to be efficient",
"That is an interesting concept",
"I will take the lead",
"Let us minimize the risk",
"This meets our requirements",
"Please keep me informed",
"We should adjust the timeline",
"That is a clear benefit",
"I understand the situation",
"Let us work together",
"This is highly effective",
"We must be consistent",
"That is a notable change",
"I will prepare the report",
"Let us reach a consensus",
"This is the standard procedure",
"Please consider the alternatives",
"We should maintain our focus",
"That is a vital point",
"I see the big picture",
"Let us proceed with caution",
"This is a productive session",
"We need to communicate better",
"That is a distinct advantage",
"I will organize the files",
"Let us evaluate the performance",
"This is a comprehensive guide",
"Please respect the deadline",
"We should optimize this process",
"That is an informed decision",
"I am confident in this",
"Let us wrap this up",
"This is the desired outcome",
"We are on the right track",
"Please look at the screen",
"Let me show you",
"Is everyone ready",
"This works perfectly",
"Next slide please",
"Notice the layout here",
"Can you see this clearly",
"This part is very important",
"Let us move on now",
"Watch how this changes",
"See the results here",
"Click the top button",
"Open the main menu",
"Select the first option",
"The system is now active",
"Check the status light",
"This is a new feature",
"Observe the data flow",
"Note the smooth transition",
"Everything is running well",
"Look at this chart",
"The graph shows the trend",
"Use the scroll bar",
"Highlight the key points",
"This is very easy",
"Follow the instructions provided",
"The interface is very clean",
"Drag the file here",
"Drop it into the folder",
"Save the changes now",
"Refresh the display",
"The process is complete",
"Zoom in on the image",
"Focus on the center",
"This button starts everything",
"Switch to the next view",
"The animation is playing",
"See the progress bar",
"It will finish soon",
"The connection is established",
"Type your text here",
"Delete the old version",
"Copy the link now",
"Paste it in the box",
"Press the enter key",
"The window is now open",
"Close the application safely",
"Restart the demonstration",
"The quality is very high",
"Look at the bottom left",
"Check the top right corner",
"This is the main dashboard",
"Explore the different settings",
"Adjust the brightness level",
"The text is very legible",
"Navigate to the home page",
"Click the red icon",
"The signal is very strong",
"This is the default state",
"Modify the current parameters",
"The update is installing",
"Wait for the signal",
"The hardware is ready",
"Plug in the cable",
"Turn the device on",
"Watch the indicator light",
"This shows the error",
"We will fix it now",
"Try the alternative method",
"The design is modern",
"See the detailed view",
"Collapse the side panel",
"Expand the search results",
"Filter by the category",
"Sort the list alphabetically",
"The preview is available",
"Print the document now",
"Export the file as PDF",
"Share the screen please",
"The presentation is starting",
"Pay attention to this",
"This is the login screen",
"Enter the security code",
"The access is granted",
"Review the summary page",
"This is the final step",
"Confirm the selection now",
"The task is successful",
"Look at the icons",
"The menu is simple",
"Swipe to the left",
"Tap the screen once",
"Hold the button down",
"Release it slowly",
"The machine is quiet",
"The speed is impressive",
"This saves a lot",
"It is very efficient",
"Any questions about this",
"Thank you for watching",
"That is amazing news",
"I am very excited",
"I am tired today",
"What is the plan",
"Let me check that",
"How are you feeling",
"It is a nice day",
"I hope you are well",
"That sounds like fun",
"When shall we meet",
"I am very happy",
"This is a surprise",
"I am quite busy",
"Let us go outside",
"Where are we going",
"I am coming soon",
"Please wait for me",
"It is getting late",
"I should go now",
"Call me later tonight",
"Send me a message",
"I will be there",
"That is very kind",
"You are very welcome",
"Thank you so much",
"I am very sorry",
"Please forgive me",
"It is quite alright",
"No problem at all",
"I am very glad",
"That is very funny",
"I am laughing now",
"Tell me a story",
"I am listening closely",
"That is very interesting",
"I want to learn",
"This lesson is great",
"I am very curious",
"What happened next",
"I am so proud",
"This is truly beautiful",
"Look at the view",
"The weather is lovely",
"It is raining outside",
"I like the sunshine",
"The clouds are pretty",
"It is very peaceful",
"I hear some music",
"That sounds very nice",
"I am so tired",
"I need some sleep",
"Good night to you",
"Have some sweet dreams",
"Wake up early tomorrow",
"It is morning already",
"The breakfast is ready",
"The tea is hot",
"I like this drink",
"It tastes very good",
"I am full now",
"Let us walk together",
"The park is beautiful",
"I see a bird",
"The trees are tall",
"Nature is very calm",
"I love this place",
"Let us stay longer",
"We should go home",
"I am almost there",
"Open the mail please",
"Read the new book",
"I like this story",
"The ending was perfect",
"I am very surprised",
"That was quite unexpected",
"I am a bit confused",
"Please explain it again",
"I understand it now",
"It is very simple",
"I am very sure",
"Maybe you are right",
"I agree with you",
"Let us try that",
"It will be fine",
"Do not worry much",
"Everything is just fine",
"I am right here",
"You are not alone",
"We are in this together",
"I support your choice",
"That is very thoughtful",
"I appreciate the help",
"You are very helpful",
"I am so lucky",
"This is a gift",
"Keep the small change",
"It is my pleasure",
"Have a wonderful time",
"Enjoy your whole day",
"I will miss you",
"I need some water",
"I am very thirsty",
"Where is the water",
"Give me a glass",
"I am hungry now",
"Where is the food",
"I want a snack",
"Please provide some food",
"I need a napkin",
"Where is the restroom",
"I need the bathroom",
"Is it located nearby",
"Please help me now",
"I am quite lost",
"Which way should I go",
"I need medical help",
"I feel very sick",
"My head is hurting",
"I need some medicine",
"Call for some assistance",
"I am very cold",
"Where is my coat",
"I need a blanket",
"Turn up the heat",
"I am very hot",
"Turn on the fan",
"Open the window please",
"I need fresh air",
"Where is my phone",
"I lost my keys",
"I need my bag",
"Pass me the remote",
"Turn on the light",
"It is too dark",
"I cannot see well",
"Where are my glasses",
"I need to sit",
"Find me a chair",
"I am very tired",
"I need a nap",
"Wake me up later",
"What time is it",
"Is it getting late",
"I need to go",
"Where is the exit",
"Open the front door",
"Lock the door now",
"I need a pen",
"Give me some paper",
"I want to write",
"Show me the way",
"Hold my hand please",
"I need a hug",
"Please stay with me",
"Do not go away",
"I am a bit afraid",
"I feel safe now",
"Thank you for helping",
"You are very kind",
"I need some advice",
"Tell me what happens",
"I am listening well",
"Speak louder please",
"I cannot hear you",
"Turn down the volume",
"It is too loud",
"I need some quiet",
"Please be very still",
"Wait a minute please",
"Slow down the pace",
"Hurry up now please",
"We are running late",
"I am ready now",
"Let us go together",
"Where is the car",
"Drive very carefully please",
"Stop the car now",
"I want to walk",
"It is very far",
"Is the place close",
"I am almost there",
"Please look at me",
"Listen to me now",
"I have a secret",
"Do not tell anyone",
"I need a friend",
"You are my friend",
"I like you a lot",
"I am very happy",
"This is absolutely perfect",
"I want this one",
"Not that one please",
"Give it to me",
"Take it away now",
"I am done now",
"Clean this mess up",
"It is very messy",
"I need a towel",
"Wash your hands please",
"Dry your face now"

]

def train_model():
    # 1. Validation
    if not YOUR_CUSTOM_DATA:
        print("ERROR: The dataset is empty!")
        print("Please paste your AI-generated sentences into the 'YOUR_CUSTOM_DATA' list in the script.")
        return

    print(f"Processing {len(YOUR_CUSTOM_DATA)} sentences...")
    
    # 2. Load or Initialize Memory
    if os.path.exists(TARGET_FILE):
        try:
            with open(TARGET_FILE, 'r') as f:
                data = json.load(f)
            print(f"Loaded existing memory ({len(data['user_words'])} words). Merging new data...")
        except:
            data = _create_empty_memory()
    else:
        data = _create_empty_memory()

    # 3. Add Core Defaults (Safety Net)
    defaults = ["HELLO", "YES", "NO", "GOOD", "BAD", "HELP", "STOP", "GO", "COME", 
                "I", "YOU", "HE", "SHE", "IT", "WE", "THEY", "THE", "AND", "TO", "A"]
    for word in defaults:
        if word not in data["core_words"]:
            data["core_words"][word] = {"source": "default"}

    # 4. THE MAGIC: Processing the Dataset
    # We set the timestamp to 30 days ago. 
    # This ensures your REAL usage (Today) always beats this synthetic data.
    OLD_TIME = time.time() - (30 * 24 * 60 * 60) 

    count_ngrams = 0
    
    for sentence in YOUR_CUSTOM_DATA:
        # Clean the sentence
        clean_sentence = " ".join(sentence.upper().strip().split())
        words = clean_sentence.split()
        
        if len(words) < 2: continue

        # A. Register Individual Words (Low Confidence)
        for w in words:
            if w not in data["user_words"]:
                data["user_words"][w] = {
                    "frequency": 1,        # Lowest possible rank
                    "last_used": OLD_TIME  # Old timestamp
                }
            # Note: We do NOT increment frequency for existing words.

        # B. Register N-Gram Context
        history = []
        for w in words:
            if len(history) > 0:
                target = w
                # Learn contexts from 1 word back up to N-1 words back
                for i in range(1, N_ORDER):
                    if len(history) < i: break
                    
                    context = " ".join(history[-i:]) 
                    
                    if context not in data["ngrams"]:
                        data["ngrams"][context] = {}
                    
                    if target not in data["ngrams"][context]:
                        data["ngrams"][context][target] = 0
                        
                    # CAP frequency at 1. 
                    if data["ngrams"][context][target] < 1:
                        data["ngrams"][context][target] = 1
                        count_ngrams += 1
            
            history.append(w)

    # 5. Save to Disk
    try:
        with open(TARGET_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print("------------------------------------------------")
        print(f"SUCCESS! Magic applied.")
        print(f"- Memory File: {TARGET_FILE}")
        print(f"- Total Words: {len(data['user_words'])}")
        print(f"- Context Rules: {len(data['ngrams'])}")
        print("------------------------------------------------")
    except Exception as e:
        print(f"Error saving file: {e}")

def _create_empty_memory():
    return {
        "core_words": {},
        "user_words": {},
        "ngrams": {},
        "stats": {"created": time.time(), "type": "custom-trained"}
    }

if __name__ == "__main__":
    train_model()