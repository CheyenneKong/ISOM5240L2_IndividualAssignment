import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
from PIL import Image
import time
import os

# --- 1. CORE AI FUNCTIONS ---

@st.cache_resource
def load_models():
    """Loads lightweight models that run fast on Streamlit Cloud."""
    # Image Captioning
    cap_model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(cap_model_id)
    caption_model = BlipForConditionalGeneration.from_pretrained(cap_model_id)
    
    # Lightweight Storyteller (DistilGPT2)
    story_gen = pipeline("text-generation", model="distilgpt2")
    
    return processor, caption_model, story_gen

processor, caption_model, story_gen = load_models()

def img2text(image):
    """Describes the image clearly for the storyteller."""
    inputs = processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    text = processor.decode(out[0], skip_special_tokens=True)
    text = text.replace("illustration", "scene").replace("drawing", "place")
    return text

def text2story(description):
    """Enhanced Storyteller for ages 3-10 with TA Guardrails."""
    prompt = f"A fun kids story about {description}. Everyone was happy and playing. Once upon a time, "
    
    story_output = story_gen(
        prompt, 
        max_new_tokens=90, 
        do_sample=True,    
        temperature=0.5,   
        top_p=0.8,        
        repetition_penalty=1.4
    )
    
    full_text = story_output[0]['generated_text']
    
    if "Once upon a time, " in full_text:
        story_only = "Once upon a time, " + full_text.split("Once upon a time, ")[-1]
    else:
        story_only = full_text

    # Vocabulary Simplification Map
    simple_vocab_map = {
        "destination": "special spot", "journey": "trip", "extremely": "very",
        "beautiful": "pretty", "adventure": "fun game", "magnificent": "great"
    }
    for complex_word, simple_word in simple_vocab_map.items():
        story_only = story_only.replace(complex_word, simple_word)

    # Safety Filter
    forbidden_words = ["death", "scary", "dangerous", "hurt", "sad", "october", "year"]
    for word in forbidden_words:
        story_only = story_only.replace(word, "magic")

    if "." in story_only:
        story_only = story_only[:story_only.rfind(".")+1]
        
    # Length Guardrail (Requirement: 50-100 words)
    words = story_only.split()
    if len(words) < 55:
        story_only += " 'Yay!' they all shouted together. It was the best day ever and everyone had a big smile on their face. The end!"
        
    return story_only

def text2audio(story_text):
    """Converts story to audio[cite: 16, 25]."""
    tts = gTTS(text=story_text, lang='en', tld='com.au')
    audio_file = "story_audio.mp3"
    tts.save(audio_file)
    return audio_file

# --- 2. THE MAGIC INTERFACE ---

def main():
    # Browser Tab Setup [cite: 28]
    st.set_page_config(page_title="Your Magic Story-Bot", page_icon="🤖")

    # --- SUGGESTION #3: Colorful Background ---
    # Custom CSS to make the app more child-friendly
    st.markdown("""
        <style>
        .stApp {
            background-color: #FDFCF0; /* Soft cream/yellow for a storybook feel */
        }
        .story-container {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            border: 3px solid #FFD700;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
            font-size: 18px;
            color: #333;
        }
        </style>
        """, unsafe_allow_html=True)

    # Rainbow Title and Header [cite: 28, 31]
    st.markdown("# :rainbow[✨ Your Magic Story-Bot ✨] 🤖")
    st.markdown("### :violet[Upload a photo and let's go on an adventure!]")
    st.markdown("---")

    # Image Input [cite: 13, 20]
    uploaded_file = st.file_uploader("📸 Pick a picture...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='🌈 Your Magical Portal', use_container_width=True)
        
        if st.button("🪄 Cast Story Spell! ✨"):
            # Magic Progress Bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.markdown("### 🧐 :orange[The Robot is putting on its glasses...]")
            description = img2text(image)
            progress_bar.progress(33)
            time.sleep(1)
            
            status_text.markdown("### ✍️ :green[Mixing colors and words...]")
            story = text2story(description)
            progress_bar.progress(66)
            time.sleep(1)
            
            status_text.markdown("### ✨ :blue[Sprinkling fairy dust...]")
            audio_path = text2audio(story)
            progress_bar.progress(100)
            time.sleep(1)
            
            progress_bar.empty()
            status_text.empty()

            # --- SUGGESTION #1: Celebration ---
            st.balloons() 

            # --- SUGGESTION #5: Storybook Card Formatting ---
            st.markdown("## :orange[📖 Your Magical Tale]")
            st.markdown(f'<div class="story-container">{story}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("## :blue[🎧 Hear the Magic]")
            st.markdown("##### 📢 Press play to hear your story, little adventurer!")
            st.audio(audio_path) # [cite: 16]
            
            st.markdown(":tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")

if __name__ == "__main__":
    main()
