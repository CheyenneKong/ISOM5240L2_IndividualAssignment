import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
from PIL import Image
import time
import os

# --- 1. CORE AI FUNCTIONS ---

@st.cache_resource
def load_models():
    """Loads lightweight models for Streamlit Cloud deployment."""
    # Image Captioning [cite: 20, 21]
    cap_model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(cap_model_id)
    caption_model = BlipForConditionalGeneration.from_pretrained(cap_model_id)
    
    # Text Generation [cite: 23]
    story_gen = pipeline("text-generation", model="distilgpt2")
    
    return processor, caption_model, story_gen

processor, caption_model, story_gen = load_models()

def img2text(image):
    """Extracts caption from image using BLIP[cite: 20]."""
    inputs = processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    text = processor.decode(out[0], skip_special_tokens=True)
    return text

def text2story(description):
    """
    Generates a 50-100 word third-person narrative[cite: 8, 14].
    """
    # Prompt priming for third-person perspective and happy tone
    prompt = f"A happy story for kids about {description}. The friends were playing. Once upon a time, they "
    
    story_output = story_gen(
        prompt, 
        max_new_tokens=90, 
        do_sample=True,    
        temperature=0.4,   
        top_p=0.85,        
        repetition_penalty=1.5
    )
    
    full_text = story_output[0]['generated_text']
    
    # Extract only the story section
    if "Once upon a time, they " in full_text:
        story_only = "Once upon a time, they " + full_text.split("Once upon a time, they ")[-1]
    else:
        story_only = full_text

    # Safety and Vocabulary Filters
    forbidden = ["death", "scary", " I ", " me ", " my ", "mine"]
    for word in forbidden:
        story_only = story_only.replace(word, "magic")

    if "." in story_only:
        story_only = story_only[:story_only.rfind(".")+1]
        
    # Word count requirement check (50-100 words) 
    words = story_only.split()
    if len(words) < 55:
        story_only += " They all smiled and shared a wonderful adventure together. It was the best day ever in their happy world! The end!"
        
    return story_only

def text2audio(story_text):
    """Converts text to audio using gTTS[cite: 25]."""
    tts = gTTS(text=story_text, lang='en', tld='com.au')
    audio_file = "story_audio.mp3"
    tts.save(audio_file)
    return audio_file

# --- 2. THE MAGIC INTERFACE ---

def main():
    st.set_page_config(page_title="Your Magic Story-Bot", page_icon="🤖")

    # --- UI UPDATE: Sky Blue Background for better readability ---
    st.markdown("""
        <style>
        .stApp {
            background-color: #E0F7FA; /* Soft Sky Blue */
        }
        .story-container {
            background-color: white;
            padding: 30px;
            border-radius: 20px;
            border: 5px solid #4FC3F7;
            font-size: 20px;
            color: #2C3E50;
            line-height: 1.6;
        }
        /* Darker text colors for readability against sky blue */
        h1, h2, h3, p, span, label {
            color: #01579B !important;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("# :rainbow[✨ Your Magic Story-Bot ✨] 🤖")
    st.markdown("### Let's find a story in your picture!")
    st.markdown("---")

    uploaded_file = st.file_uploader("📸 Upload your photo here:", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='🌈 Your Magical Portal', use_container_width=True)
        
        if st.button("🪄 Cast Story Spell! ✨"):
            progress_bar = st.progress(0)
            
            # Step 1: Image to Text [cite: 13]
            st.write("🧐 **The Robot is looking at your photo...**")
            description = img2text(image)
            progress_bar.progress(33)
            
            # Step 2: Text to Story 
            st.write("✍️ **Writing a happy adventure...**")
            story = text2story(description)
            progress_bar.progress(66)
            
            # Step 3: Text to Audio [cite: 16]
            st.write("✨ **Adding the magic voice...**")
            audio_path = text2audio(story)
            progress_bar.progress(100)
            
            st.balloons() 

            st.markdown("## 📖 Your Magical Tale")
            st.markdown(f'<div class="story-container">{story}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("## 🎧 Hear the Magic")
            st.audio(audio_path) 
            
            st.markdown("🌟 🎈 🎨 🍦 🍭 🎠")

if __name__ == "__main__":
    main()
