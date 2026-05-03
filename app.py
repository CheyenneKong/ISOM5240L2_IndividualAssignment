import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
from PIL import Image
import time
import os

# --- 1. CORE AI FUNCTIONS ---

@st.cache_resource
def load_models():
    """Pre-loads both models to save time later."""
    # Image Captioning
    cap_model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(cap_model_id)
    caption_model = BlipForConditionalGeneration.from_pretrained(cap_model_id)
    
    # Storyteller - Switched to TinyLlama for better kid stories and no "-sama" errors
    story_gen = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    return processor, caption_model, story_gen

processor, caption_model, story_gen = load_models()

def img2text(image):
    """Describes the image and removes technical jargon."""
    inputs = processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    text = processor.decode(out[0], skip_special_tokens=True)
    # Clean-up technical words
    text = text.replace("illustration", "scene").replace("drawing", "place")
    return text

def text2story(description):
    """Creates a concrete 50-100 word story for kids."""
    # Strict prompt to prevent weird repetitions
    prompt = f"<|system|>\nYou are a friendly children's storyteller. Write a 60-word happy story based on this: {description}.<|user|>\nOnce upon a time,"
    
    story_output = story_gen(prompt, max_new_tokens=100, do_sample=True, temperature=0.7, top_k=50)
    
    # Cleaning the output to remove the 'system' instructions
    full_text = story_output[0]['generated_text']
    story_only = full_text.split("<|user|>\n")[-1]
    
    # Ensure it ends at a full stop
    if "." in story_only:
        story_only = story_only[:story_only.rfind(".")+1]
    return story_only

def text2audio(story_text):
    """Converts story to a friendly voice."""
    tts = gTTS(text=story_text, lang='en', tld='com.au')
    audio_file = "story_audio.mp3"
    tts.save(audio_file)
    return audio_file

# --- 2. RAINBOW USER INTERFACE ---

def main():
    st.set_page_config(page_title="Magic Storybook", page_icon="🦄")

    st.markdown("# :rainbow[✨ My Magic AI Storybook ✨]")
    st.markdown("### :violet[Upload a photo to see the magic!]")

    uploaded_file = st.file_uploader("📸 Pick a picture...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='🌈 Your Magical Portal', use_container_width=True)
        
        if st.button("🪄 Cast Story Spell! ✨"):
            # 1. Kid-friendly Artificial Delay
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            messages = ["🔍 Finding the magic...", "✨ Sprinkling fairy dust...", "📖 Writing the adventure..."]
            for i in range(3):
                status_text.markdown(f"### :orange[{messages[i]}]")
                for percent in range(i*33, (i+1)*33):
                    time.sleep(0.03) # Total roughly 3 seconds
                    progress_bar.progress(percent + 1)
            
            # 2. Run the actual AI magic
            description = img2text(image)
            story = text2story(description)
            audio_path = text2audio(story)
            
            progress_bar.empty()
            status_text.empty()

            # 3. Show the final results
            st.markdown("## :orange[📖 Your Magical Tale]")
            st.info(story)
            
            st.markdown("## :blue[🎧 Hear the Magic]")
            st.audio(audio_path)
            
            st.markdown(":tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")

if __name__ == "__main__":
    main()
