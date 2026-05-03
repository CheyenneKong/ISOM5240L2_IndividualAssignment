import streamlit as st
from transformers import pipeline
from gtts import gTTS
from PIL import Image
import os

# --- 1. THE FUNCTIONS (The Factory Stations) ---

def img2text(url):
    """Turns an image into a simple text description."""
    # Using the specific model recommended in the assignment guide
    image_to_text_model = pipeline("image-captioning", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

def text2story(text):
    """Expands a short description into a 50-100 word story."""
    # We use a text-generation model (GPT-2) to 'imagine' a story
    story_generator = pipeline("text-generation", model="gpt2")
    
    # We create a prompt to tell the AI it's writing for kids
    prompt = f"Once upon a time, there was {text}. It was a beautiful day and"
    
    # Generate text: max_new_tokens ensures we hit that 50-100 word requirement
    story_output = story_generator(prompt, max_new_tokens=80, num_return_sequences=1)
    return story_output[0]['generated_text']

def text2audio(story_text):
    """Converts the written story into an MP3 audio file."""
    # gTTS is a simple way to create speech for the 'audio experience' requirement
    tts = gTTS(text=story_text, lang='en')
    audio_file = "story_audio.mp3"
    tts.save(audio_file)
    return audio_file

# --- 2. THE MAIN PART (The User Interface) ---

def main():
    st.set_page_config(page_title="Kids Storyteller", page_icon="📖")
    st.header("📖 AI Storyteller for Kids")
    st.write("Upload a photo and listen to a magical story!")

    # Image upload input
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the image the user uploaded
        image = Image.open(uploaded_file)
        st.image(image, caption='Your Uploaded Image', use_container_width=True)
        
        # When the user clicks the button, trigger the pipeline
        if st.button("Generate Magic"):
            with st.spinner('AI is thinking...'):
                # Step 1: Image to Text
                description = img2text(image)
                
                # Step 2: Text to Story
                story = text2story(description)
                
                # Step 3: Story to Audio
                audio_path = text2audio(story)
                
                # Show results in the UI
                st.subheader("The Story")
                st.write(story)
                
                st.subheader("Listen to the Story")
                st.audio(audio_path)

# This tells Python to run the main function
if __name__ == "__main__":
    main()
