import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv
import io
from groq import Groq
import base64

# Load environment variables
load_dotenv()

# Configure API keys
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Please set GEMINI_API_KEY in .env file")
    st.stop()

if not GROQ_API_KEY:
    st.error("Please set GROQ_API_KEY in .env file")
    st.stop()

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Configure Groq
groq_client = Groq(api_key=GROQ_API_KEY)

# Define available models
GEMINI_MODELS = {
    "Gemini 2.0 Flash": "gemini-2.0-flash",
    "Gemini 2.0 Flash Lite": "gemini-2.0-flash-lite-preview-02-05",
    "Gemini 1.5 Flash": "gemini-1.5-flash",
    "Gemini 1.5 Flash 8B": "gemini-1.5-flash-8b"
}

LLAMA_MODELS = {
    "Llama 3.2 90B Vision": "llama-3.2-90b-vision-preview",
    "Llama 3.2 11B Vision": "llama-3.2-11b-vision-preview"
}

# Configure Streamlit page
st.set_page_config(
    page_title="Math Question Extractor",
    page_icon="📝",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    .model-selector {
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def get_gemini_response(input_image, model_name):
    """Extract math question from image using selected Gemini model"""
    model = genai.GenerativeModel(model_name)
    prompt = "extract what exactly in the image. no preamble "
    response = model.generate_content([prompt, input_image])
    return response.text

def get_llama_response(input_image, model_name):
    """Extract math question from image using selected Llama model"""
    base64_image = encode_image_to_base64(input_image)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "extract the complete question, how it was arranged, do not explain, no preamble "
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    completion = groq_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        top_p=1,
        stream=False
    )
    
    return completion.choices[0].message.content

def load_image():
    """Image loader and processor"""
    uploaded_file = st.file_uploader(
        "Upload an image containing a math question", 
        type=['png', 'jpg', 'jpeg'],
        help="Supported formats: PNG, JPG, JPEG"
    )
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))
        return image
    return None

def main():
    st.title("📝 Math Question Extractor")
    st.markdown("---")

    # Model selection in sidebar
    st.sidebar.header("Model Selection")
    
    # Model type selection (Gemini or Llama)
    model_type = st.sidebar.radio(
        "Select Model Type",
        ["Gemini", "Llama"],
        help="Choose between Gemini and Llama model families"
    )
    
    # Model selection based on type
    if model_type == "Gemini":
        selected_model_name = st.sidebar.selectbox(
            "Select Gemini Model",
            options=list(GEMINI_MODELS.keys()),
            index=0,
            help="Choose the Gemini model you want to use for extraction",
            key="gemini_model"
        )
        selected_model = GEMINI_MODELS[selected_model_name]
    else:
        selected_model_name = st.sidebar.selectbox(
            "Select Llama Model",
            options=list(LLAMA_MODELS.keys()),
            index=0,
            help="Choose the Llama model you want to use for extraction",
            key="llama_model"
        )
        selected_model = LLAMA_MODELS[selected_model_name]
    
    # Display selected model
    st.sidebar.markdown(f"**Selected Model:** {selected_model_name}")
    st.sidebar.markdown("---")

    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Image")
        image = load_image()
        if image:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Extract button
            if st.button("Extract Question", type="primary"):
                with st.spinner(f"Extracting question using {selected_model_name}..."):
                    try:
                        with col2:
                            st.markdown("### Extracted Question")
                            
                            # Use appropriate model based on selection
                            if model_type == "Gemini":
                                question = get_gemini_response(image, selected_model)
                            else:
                                question = get_llama_response(image, selected_model)
                            
                            st.write(question)
                            
                            # Download option
                            st.download_button(
                                label="Download Question",
                                data=question,
                                file_name="extracted_question.txt",
                                mime="text/plain"
                            )
                    except Exception as e:
                        st.error(f"Error during extraction: {str(e)}")

    # App info
    with st.expander("About"):
        st.markdown("""
            Upload an image containing a mathematics question to extract just the question text.
            The app uses AI to identify and extract only the mathematical question from the image.
            
            Available Models:
            
            Gemini Models:
            - Gemini 2.0 Flash: Latest version, optimized for quick responses
            - Gemini 2.0 Flash Lite: Lightweight version of 2.0 Flash
            - Gemini 1.5 Flash: Balanced performance and speed
            - Gemini 1.5 Flash 8B: Efficient 8B parameter version
            
            Llama Models:
            - Llama 3.2 90B Vision: Large vision model with 90B parameters
            - Llama 3.2 11B Vision: Efficient vision model with 11B parameters
        """)

if __name__ == "__main__":
    main()
