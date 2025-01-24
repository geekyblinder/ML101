import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import google.generativeai as genai
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['GRPC_VERBOSITY'] = 'NONE'

image = Image.open("x.jpg").convert("RGB")

def parse_response(response):
    candidates = response['result']['candidates']  # Accessing candidates list
    text = candidates[0]['content']['parts'][0]['text']  # Accessing the text content
    return text


def clip_sketch_classifier():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    text = ["a handmade sketch", "not a handmade sketch"]
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True).to(device)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    if probs[0][0] > probs[0][1]:
        return True  # It is a handmade sketch
    return False  # It is not a handmade sketch

def blip_gemini_pipeline():
    torch.set_num_threads(4)

    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs)
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)

    api_key = os.getenv("GENAI_API_KEY")  # Secure API key storage
    genai.configure(api_key=api_key)

    try:
        api_key = 'API_KEY'
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("This is the description of an sktech which is still at an ideation phase - " + caption + " how can I improve this sketch to make it beautiful?")
        # text = response["candidates"][0]["content"]["parts"][0]["text"]
        # print(response)
        # print(parse_response(response))
        try:
            candidates = response.candidates
            text = candidates[0].content.parts[0].text
            print(text)
        except AttributeError as e:
            print(f"Error accessing response: {e}")

    except Exception as e:
        print(f"Error while generating content from Gemini API: {e}")

def __main__():
    if clip_sketch_classifier():
        print("It is a sketch")
        blip_gemini_pipeline()
    else:
        print("Not a sketch. Please upload a sketch.")

if __name__ == "__main__":
    __main__()