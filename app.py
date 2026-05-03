import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
from PIL import Image
import os

# --- 1. THE FUNCTIONS ---

@st.cache_resource
def load_captioning_model():
    """Bypasses the 'pipeline' task error by loading components directly."""
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    return processor, model

def img2text(image):
    """Turns an image into a simple text description."""
    processor, model = load_captioning_model()
    # Prepare image for the model
    inputs = processor(image, return_tensors="pt")
    # Generate the caption
    out = model.generate(**inputs)
    # Convert output tokens back to text
    text = processor.decode(out[0], skip_special_tokens=True)
    return text

def text2story(text):
    """Expands a short description into a 50-100 word story."""
    # We keep the pipeline for text-generation as 'gpt2' is more standard
    story_generator = pipeline("text-generation", model="gpt2")
    prompt = f"Once upon a time, there was {text}. It was a beautiful day and"
    story_output = story_generator(prompt, max_new_tokens=80, num_return_sequences=1)
    return story_output[0]['generated_text']

def text2audio(story_text):
    """Converts the written story into an MP3 audio file."""
    tts = gTTS(text=story_text, lang='en')
    audio_file = "story_audio.mp3"
    tts.save(audio_file)
    return audio_file

# --- 2. THE MAIN PART ---

def main():
    st.set_page_config(page_title="Kids Storyteller", page_icon="📖")
    st.header("📖 AI Storyteller for Kids")
    st.write("Upload a photo and listen to a magical story!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Your Uploaded Image', use_container_width=True)
        
        if st.button("Generate Magic"):
            with st.spinner('AI is thinking...'):
                # Step 1: Image to Text
                description = img2text(image)
                
                # Step 2: Text to Story
                story = text2story(description)
                
                # Step 3: Story to Audio
                audio_path = text2audio(story)
                
                st.subheader("The Story")
                st.write(story)
                
                st.subheader("Listen to the Story")
                st.audio(audio_path)

if __name__ == "__main__":
    main()
