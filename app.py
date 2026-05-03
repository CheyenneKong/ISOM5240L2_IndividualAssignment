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
    
    # Lightweight Storyteller (DistilGPT2 is faster and uses less memory)
    story_gen = pipeline("text-generation", model="distilgpt2")
    
    return processor, caption_model, story_gen

processor, caption_model, story_gen = load_models()

def img2text(image):
    """Describes the image clearly for the storyteller."""
    inputs = processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    text = processor.decode(out[0], skip_special_tokens=True)
    # Removing technical words that confuse the story
    text = text.replace("illustration", "scene").replace("drawing", "place")
    return text

def text2story(description):
    """Creates a joyful 50-100 word story specifically for kids."""
    # We add 'positive' keywords to the prompt to block scary words like 'dangerous'
    prompt = (
        f"Write a cheerful, happy children's story about {description}. "
        "The sun was shining and all the friends were laughing. "
        "Once upon a time,"
    )
    
    # max_new_tokens set to 80 to ensure we meet the 50-100 word requirement
    story_output = story_gen(
        prompt, 
        max_new_tokens=80, 
        do_sample=True, 
        temperature=0.7, # Lower temperature makes the AI less 'wild' and more focused
        top_k=40,
        repetition_penalty=1.2 # This stops it from saying 'jumping, jumping, jumping'
    )
    
    full_text = story_output[0]['generated_text']
    
    # Remove the instruction part so the kids only see the story
    story_only = full_text.split("Once upon a time,")[-1]
    final_story = "Once upon a time," + story_only

    # Extra safety: Clean out any weird words just in case
    scary_words = ["dangerous", "scary", "closed", "monster", "animal"]
    for word in scary_words:
        final_story = final_story.replace(word, "magical")

    # Cut off at the last complete sentence for a professional finish
    if "." in final_story:
        final_story = final_story[:final_story.rfind(".")+1]
        
    return final_story

def text2audio(story_text):
    """Fast audio generation for the 'audio format' requirement."""
    tts = gTTS(text=story_text, lang='en', tld='com.au')
    audio_file = "story_audio.mp3"
    tts.save(audio_file)
    return audio_file

# --- 2. THE RAINBOW INTERFACE ---

def main():
    st.set_page_config(page_title="Your Magic Story-Bot", page_icon="🤖")

    # Rainbow Title
    st.markdown("# :rainbow[✨ Your Magic Story-Bot ✨] 🤖")
    st.markdown("### :violet[Upload a photo and let's go on an adventure!]")
    st.markdown("---")

    uploaded_file = st.file_uploader("📸 Pick a picture...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='🌈 Your Magical Portal', use_container_width=True)
        
        if st.button("🪄 Cast Story Spell! ✨"):
            # Magic Progress Bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Descibe
            status_text.markdown("### :orange[🔍 Seeing the magic...]")
            description = img2text(image)
            progress_bar.progress(33)
            time.sleep(1) # Visual pause for the '3 second' effect
            
            # Step 2: Story
            status_text.markdown("### :green[📖 Writing the adventure...]")
            story = text2story(description)
            progress_bar.progress(66)
            time.sleep(1)
            
            # Step 3: Audio
            status_text.markdown("### :blue[✨ Sprinkling fairy dust...]")
            audio_path = text2audio(story)
            progress_bar.progress(100)
            time.sleep(1)
            
            # Clear loading UI
            progress_bar.empty()
            status_text.empty()

            # Final Results
            st.markdown("## :orange[📖 Your Magical Tale]")
            st.info(story)
            
            st.markdown("## :blue[🎧 Hear the Magic]")
            st.audio(audio_path)
            
            st.markdown(":tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")

if __name__ == "__main__":
    main()
