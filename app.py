import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
from PIL import Image
import time
import os

# --- SECTION 1: MODEL INITIALIZATION ---

@st.cache_resource
def load_models():
    """
    Loads and caches the Deep Learning models to optimize performance 
    on Streamlit Community Cloud.
    """
    # Image Captioning Model: BLIP (Bootstrapping Language-Image Pre-training)
    cap_model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(cap_model_id)
    caption_model = BlipForConditionalGeneration.from_pretrained(cap_model_id)
    
    # Story Generation Model: DistilGPT2 (Lightweight version of GPT-2)
    story_gen = pipeline("text-generation", model="distilgpt2")
    
    return processor, caption_model, story_gen

processor, caption_model, story_gen = load_models()

# --- SECTION 2: AI LOGIC FUNCTIONS ---

def img2text(image):
    """Converts uploaded image pixels into a descriptive text caption."""
    inputs = processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    text = processor.decode(out[0], skip_special_tokens=True)
    return text

def text2story(description):
    """
    Expands the image caption into a child-friendly story (50-100 words).
    Includes safety filters to ensure content is appropriate for ages 3-10.
    """
    # Prompt engineering to guide the model toward an innocent narrative
    prompt = f"Write a happy, innocent story for young children about {description}. The animal friends were playing. Once upon a time, they "
    
    story_output = story_gen(
        prompt, 
        max_new_tokens=90, 
        do_sample=True,    
        temperature=0.4,   
        top_p=0.85,        
        repetition_penalty=1.5,
        early_stopping=True
    )
    
    full_text = story_output[0]['generated_text']
    
    # Extract only the narrative portion
    if "Once upon a time, they " in full_text:
        story_only = "Once upon a time, they " + full_text.split("Once upon a time, they ")[-1]
    else:
        story_only = full_text

    # Safety Filter: Replaces adult-themed vocabulary with kid-friendly alternatives
    forbidden = [
        "death", "scary", " I ", " me ", " my ", "weed", "pot", "smoke", 
        "drug", "alcohol", "drink", "beer", "wine", "sex", "naked", "18+"
    ]
    for word in forbidden:
        if word.lower() in story_only.lower():
            story_only = story_only.replace(word, "magic rainbows")

    # Clean up sentence endings
    if "." in story_only:
        story_only = story_only[:story_only.rfind(".")+1]
        
    # Minimum word count logic to meet the 50-word requirement
    words = story_only.split()
    if len(words) < 55:
        story_only += " They all shared a wonderful adventure together. It was the best day ever in their happy world! The end!"
        
    return story_only

def text2audio(story_text):
    """Converts the final narrative into an MP3 audio file using gTTS."""
    try:
        tts = gTTS(text=story_text, lang='en', tld='com.au')
        audio_file = "story_audio.mp3"
        tts.save(audio_file)
        return audio_file
    except Exception:
        return None

# --- SECTION 3: USER INTERFACE (STREAMLIT) ---

def main():
    st.set_page_config(page_title="Your Magic Story-Bot", page_icon="🤖")

    # Custom CSS for theme styling and visual engagement
    st.markdown("""
        <style>
        .stApp { background-color: #E0F7FA; }
        .story-container {
            background-color: white;
            padding: 30px;
            border-radius: 20px;
            border: 5px solid #4FC3F7;
            font-size: 20px;
            color: #2C3E50;
        }
        h1, h2, h3, p, label { color: #01579B !important; }
        .stFileUploader label {
            font-size: 26px !important;
            font-weight: bold !important;
        }
        div.stButton > button:first-child {
            background-color: #FFD700;
            color: #01579B;
            font-size: 24px;
            font-weight: bold;
            border-radius: 12px;
            border: 2px solid #01579B;
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
            # UI Feedback during processing
            progress_bar = st.progress(0)
            status_text = st.empty()
            
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

            # Final Results Display
            st.markdown("## 📖 Your Magical Tale")
            st.markdown(f'<div class="story-container">{story}</div>', unsafe_allow_html=True)
            
            st.markdown('<hr style="border: 1px solid #01579B;">', unsafe_allow_html=True)
            st.markdown("## 🎧 Hear the Magic")
            if audio_path:
                st.audio(audio_path) 
            
            st.markdown("🌟 🎈 🎨 🍦 🍭 🎠")

if __name__ == "__main__":
    main()
