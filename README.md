# Image Captioning using BLIP (Pre-trained Model)
# Install dependencies: pip install -r requirements.txt

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import sys
import os

def generate_caption(image_path_or_url):
    # Load pre-trained model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load image (from local path or URL)
    if image_path_or_url.startswith("http"):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw)
    else:
        if not os.path.exists(image_path_or_url):
            raise FileNotFoundError(f"Image not found: {image_path_or_url}")
        image = Image.open(image_path_or_url)

    # Process image
    inputs = processor(image, return_tensors="pt")

    # Generate caption
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python caption.py <image_path_or_url>")
    else:
        image_input = sys.argv[1]
        caption = generate_caption(image_input)
        print("🖼️ Generated Caption:", caption)
