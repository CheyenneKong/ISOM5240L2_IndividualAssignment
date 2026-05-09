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
    
    # Story Generation [cite: 23]
    story_gen = pipeline("text-generation", model="distilgpt2")
    
    return processor, caption_model, story_gen

processor, caption_model, story_gen = load_models()

def img2text(image):
    """Processes uploaded image to extract details."""
    inputs = processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    text = processor.decode(out[0], skip_special_tokens=True)
    return text

def text2story(description):
    """Generates a 50-100 word third-person narrative[cite: 8, 14]."""
    # Priming the AI for third-person and a happy tone for 3-10 year olds [cite: 8]
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
    
    # Extract only the story section starting from the third-person prime
    if "Once upon a time, they " in full_text:
        story_only = "Once upon a time, they " + full_text.split("Once upon a time, they ")[-1]
    else:
        story_only = full_text

    # Safety and Vocabulary Filters for 3-10 year olds [cite: 8]
    forbidden = ["death", "scary", " I ", " me ", " my "]
    for word in forbidden:
        story_only = story_only.replace(word, "magic")

    if "." in story_only:
        story_only = story_only[:story_only.rfind(".")+1]
        
    # Ensuring word count is between 50-100 [cite: 14]
    words = story_only.split()
    if len(words) < 55:
        story_only += " They all shared a wonderful adventure together. It was the best day ever in their happy world! The end!"
        
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

    # CSS for Sky Blue theme and Storybook container [cite: 28, 31]
    st.markdown("""
        <style>
        .stApp {
            background-color: #E0F7FA; 
        }
        .story-container {
            background-color: white;
            padding: 30px;
            border-radius: 20px;
            border: 5px solid #4FC3F7;
            font-size: 20px;
            color: #2C3E50;
        }
        h1, h2, h3, p, label {
            color: #01579B !important;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("# :rainbow[✨ Your Magic Story-Bot ✨] 🤖")
    st.markdown("### Upload a photo and let's go on an adventure!")
    st.markdown('<hr style="border: 1px solid #01579B;">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📸 Pick a picture...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='🌈 Your Magical Portal', use_container_width=True)
        
        if st.button("🪄 Cast Story Spell! ✨"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # One-line big loading text 
            status_text.markdown("## 🔍 Seeing the magic...")
            description = img2text(image)
            progress_bar.progress(33)
            time.sleep(1)
            
            status_text.markdown("## 📖 Writing the adventure...")
            story = text2story(description)
            progress_bar.progress(66)
            time.sleep(1)
            
            status_text.markdown("## ✨ Sprinkling fairy dust...")
            audio_path = text2audio(story)
            progress_bar.progress(100)
            time.sleep(1)
            
            progress_bar.empty()
            status_text.empty()

            st.balloons() 

            st.markdown("## 📖 Your Magical Tale")
            st.markdown(f'<div class="story-container">{story}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("## 🎧 Hear the Magic")
            st.audio(audio_path) 
            
            st.markdown("🌟 🎈 🎨 🍦 🍭 🎠")

if __name__ == "__main__":
    main()
