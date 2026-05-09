import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
from PIL import Image
import time
import os

# --- 1. CORE AI FUNCTIONS ---

@st.cache_resource
def load_models():
    """Loads lightweight models for Streamlit Cloud deployment[cite: 7]."""
    # Image Captioning [cite: 20]
    cap_model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(cap_model_id)
    caption_model = BlipForConditionalGeneration.from_pretrained(cap_model_id)
    
    # Storyteller (DistilGPT2) [cite: 23]
    story_gen = pipeline("text-generation", model="distilgpt2")
    
    return processor, caption_model, story_gen

processor, caption_model, story_gen = load_models()

def img2text(image):
    """Processes uploaded image to extract details[cite: 13, 15]."""
    inputs = processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    text = processor.decode(out[0], skip_special_tokens=True)
    # Cleaning caption for better story flow
    text = text.replace("illustration", "scene").replace("drawing", "place")
    return text

def text2story(description):
    """
    Generates a 50-100 word third-person narrative.
    Logic: Uses 'Tone Anchoring' and 'Third-Person Priming'.
    """
    # --- CHANGE: Third-Person Priming ---
    # We force the AI to start with 'The friends' to ensure a third-person narrative.
    prompt = f"A happy story for kids about {description}. The friends were having fun. Once upon a time, they "
    
    story_output = story_gen(
        prompt, 
        max_new_tokens=90, 
        do_sample=True,    
        temperature=0.4,   
        top_p=0.85,        
        repetition_penalty=1.5
    )
    
    full_text = story_output[0]['generated_text']
    
    # Extracting the narrative [cite: 15]
    if "Once upon a time, they " in full_text:
        story_only = "Once upon a time, they " + full_text.split("Once upon a time, they ")[-1]
    else:
        story_only = full_text

    # --- Vocabulary Simplification & Safety Guards ---
    simple_vocab_map = {"destination": "spot", "extremely": "very", "beautiful": "pretty"}
    for complex_word, simple_word in simple_vocab_map.items():
        story_only = story_only.replace(complex_word, simple_word)

    forbidden_words = ["death", "scary", "dangerous", "hurt", " I ", " me ", " my "]
    for word in forbidden_words:
        story_only = story_only.replace(word, "magic")

    if "." in story_only:
        story_only = story_only[:story_only.rfind(".")+1]
        
    # --- Length Guardrail (Ensuring 50-100 words) [cite: 14] ---
    words = story_only.split()
    if len(words) < 55:
        story_only += " They all laughed and played together in the sunshine. It was the most wonderful day that the friends had ever shared. The end!"
        
    return story_only

def text2audio(story_text):
    """Converts the narrative to audio format."""
    tts = gTTS(text=story_text, lang='en', tld='com.au')
    audio_file = "story_audio.mp3"
    tts.save(audio_file)
    return audio_file

# --- 2. THE MAGIC INTERFACE ---

def main():
    st.set_page_config(page_title="Your Magic Story-Bot", page_icon="🤖")

    # --- CHANGE: Dark Pink Background ---
    st.markdown("""
        <style>
        .stApp {
            background-color: #FF1493; /* Deep Pink */
        }
        .story-container {
            background-color: white;
            padding: 25px;
            border-radius: 20px;
            border: 4px solid #FFD700;
            font-size: 18px;
            color: #333;
        }
        h1, h2, h3, p {
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("# :rainbow[✨ Your Magic Story-Bot ✨] 🤖")
    st.markdown("### Upload a photo and let's go on an adventure!")
    st.markdown("---")

    uploaded_file = st.file_uploader("📸 Pick a picture...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='🌈 Your Magical Portal', use_container_width=True)
        
        if st.button("🪄 Cast Story Spell! ✨"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.markdown("### 🧐 The Robot is putting on its glasses...")
            description = img2text(image)
            progress_bar.progress(33)
            time.sleep(1)
            
            status_text.markdown("### ✍️ Mixing colors and words...")
            story = text2story(description)
            progress_bar.progress(66)
            time.sleep(1)
            
            status_text.markdown("### ✨ Sprinkling fairy dust...")
            audio_path = text2audio(story)
            progress_bar.progress(100)
            time.sleep(1)
            
            progress_bar.empty()
            status_text.empty()

            # Balloon Effect [cite: 31]
            st.balloons() 

            # Storybook Output [cite: 8, 31]
            st.markdown("## 📖 Your Magical Tale")
            st.markdown(f'<div class="story-container">{story}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("## 🎧 Hear the Magic")
            st.markdown("##### 📢 Press play to hear the story, little adventurer!")
            st.audio(audio_path) 
            
            st.markdown("🌷 🌸 🌹 🌺 🌻 🌼")

if __name__ == "__main__":
    main()
