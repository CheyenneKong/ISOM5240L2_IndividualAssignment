# ✨ Your Magic Story-Bot 🤖
**ISOM5240 Individual Assignment - Deep Learning Business Applications**

## 🔗 Live Application
[👉 Click here to launch the Magic Story-Bot](https://isom5240l2-assignment-cheyennekong.streamlit.app/)

---

## 📖 Business Objective
The goal of this application is to provide an interactive, AI-driven educational tool for children aged 3–10. By combining Computer Vision and Natural Language Processing, the app transforms personal photos into immersive, safe, and happy narratives to encourage childhood creativity and digital engagement.

## 🛠️ Technical Workflow
This app utilizes a multi-stage Deep Learning pipeline:
1. **Image Captioning:** Uses `Salesforce/blip-image-captioning-base` to analyze the visual components of an uploaded photo.
2. **Story Generation:** Uses `DistilGPT2` with custom prompt engineering and a dedicated safety filter to generate a child-friendly narrative (50–100 words).
3. **Text-to-Speech:** Converts the generated story into audio format using `gTTS` for an accessible, multisensory experience.

## 🚀 How to Run Locally
1. Clone this repository:
   ```bash
   git clone [https://github.com/CheyenneKong/ISOM5240L2_IndividualAssignment.git](https://github.com/CheyenneKong/ISOM5240L2_IndividualAssignment.git)
Install dependencies:

Bash
pip install -r requirements.txt
Run the app:

Bash
streamlit run app.py
📦 Dependencies
The following libraries are required (and listed in the requirements.txt file):

streamlit

transformers

torch

Pillow

gTTS

🛡️ Academic Integrity
Code logic and modularity were developed in collaboration with Gemini (AI) to ensure adherence to ISOM5240 technical requirements and deployment best practices. All final implementation, safety logic, and model fine-tuning were managed and reviewed by the student.
