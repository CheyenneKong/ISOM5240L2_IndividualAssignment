import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
from PIL import Image
import os

# --- 1. CORE AI FUNCTIONS ---

@st.cache_resource
def load_captioning_model():
    """Loads the 'eyes' of our AI."""
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    return processor, model

def img2text(image):
    """Turns the picture into a simple sentence, cleaning up technical words."""
    processor, model = load_captioning_model()
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    text = processor.decode(out[0], skip_special_tokens=True)
    
    # Clean up technical words for a better story start
    text = text.replace("illustration", "magical scene").replace("drawing", "wonderland")
    return text

def text2story(text):
    """Turns the sentence into a 50-100 word adventure for kids."""
    story_generator = pipeline("text-generation", model="gpt2")
    
    # A friendly prompt to guide the AI
    prompt = f"In a far away land, there was {text}. Suddenly, a magical adventure began! "
    
    # max_new_tokens helps hit the 50-100 word assignment requirement
    story_output = story_generator(prompt, max_new_tokens=90, do_sample=True, temperature=0.8)
    
    final_story = story_output[0]['generated_text']
    
    # Ensure the story ends nicely at a full stop
    if "." in final_story:
        final_story = final_story[:final_story.rfind(".")+1]
    return final_story

def text2audio(story_text):
    """Converts the story into a friendly voice."""
    # Using the Australian 'com.au' accent for a clear, storytelling tone
    tts = gTTS(text=story_text, lang='en', tld='com.au')
    audio_file = "story_audio.mp3"
    tts.save(audio_file)
    return audio_file

# --- 2. RAINBOW USER INTERFACE ---

def main():
    st.set_page_config(page_title="Magic Storybook", page_icon="🦄")

    # Using your reference style for a colorful header!
    st.markdown("# :rainbow[✨ My Magic AI Storybook ✨]")
    
    st.markdown('''
        :red[Upload] :orange[a] :green[photo] :blue[and] :violet[watch] 
        the :rainbow[magic] happen!''')
    
    st.markdown("---")

    # Image upload input for the kids
    uploaded_file = st.file_uploader("📸 Pick a picture from your computer...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='🌈 Your Magical Portal', use_container_width=True)
        
        # Friendly button text
        if st.button("🪄 Cast Story Spell! ✨"):
            with st.spinner('🌟 Sprinkling fairy dust...'):
                
                # The 3-Step Pipeline
                description = img2text(image)
                story = text2story(description)
                audio_path = text2audio(story)
                
                # Display results with fun formatting
                st.markdown("## :orange[📖 Your Magical Tale]")
                st.write(story)
                
                st.markdown("## :blue[🎧 Hear the Magic]")
                st.audio(audio_path)
                
                # A little fun at the end
                st.markdown(":tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")

if __name__ == "__main__":
    main()
