# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
from typing import Optional
import torch
from PIL import Image

# Set environment variables for Hugging Face cache
MODEL_CACHE = "model_cache"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


from cog import BasePredictor, Input, Path, ConcatenateIterator

# Import necessary components from kimi_vl library
# Note: Assuming the kimi_vl package is in the build context
from kimi_vl.serve.inference import load_model, kimi_vl_generate
from kimi_vl.serve.chat_utils import (
    Conversation,
    IMAGE_TOKEN,
    convert_conversation_to_prompts,
    get_conv_template
)
from kimi_vl.serve.utils import strip_stop_words

# --- Cog Cache Download Logic --- GITHUB BOT EDIT
BASE_URL = f"https://weights.replicate.delivery/default/kimi-vl-a3b-thinking/{MODEL_CACHE}/"

def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        # --- Cog Cache Download --- GITHUB BOT EDIT
        print("Ensuring model cache exists...")
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
        
        model_files = [
            "models--moonshotai--Kimi-VL-A3B-Thinking.tar",
            "modules.tar",
        ]
        
        print("Checking for model files...")
        for model_file in model_files:
            url = BASE_URL + model_file
            extracted_dir_name = model_file.replace(".tar", "")
            dest_path = os.path.join(MODEL_CACHE, extracted_dir_name)
            print(f"Checking for: {dest_path}")
            if not os.path.exists(dest_path):
                print(f"Directory {extracted_dir_name} not found, downloading {model_file}...")
                download_weights(url, dest_path) 
            else:
                print(f"Found {extracted_dir_name}.")
        print("Model files download check complete.")
        # --- End Cog Cache Download --- GITHUB BOT EDIT

        print("Loading Kimi-VL model from cache...")
        model_path = "moonshotai/Kimi-VL-A3B-Thinking"
        self.model, self.processor = load_model(model_path)
        print("Model loaded successfully.")

        self.conv_template = get_conv_template("kimi-vl")


    def predict(
        self,
        prompt: str = Input(description="Text prompt for the model"),
        image: Path = Input(description="Optional image input", default=None),
        top_p: float = Input(description="Top-p sampling probability", ge=0.0, le=1.0, default=1.0),
        temperature: float = Input(description="Sampling temperature", ge=0.0, le=2.0, default=0.6),
        max_length_tokens: int = Input(
            description="Maximum number of tokens to generate", ge=1, default=2048
        ),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the Kimi-VL model"""

        pil_image = None
        if image:
            try:
                pil_image = Image.open(str(image)).convert("RGB")
                print(f"Loaded image: {image}")
            except Exception as e:
                print(f"Error loading image: {e}")
                yield f"Error: Could not load image {image}. Proceeding without image."
                pil_image = None 

        conv = self.conv_template.copy()

        user_input = prompt
        if pil_image:
            # Structure message as tuple (text, [image]) if image exists
            # Prepend IMAGE_TOKEN for model processing
            user_input = (IMAGE_TOKEN + '\n' + prompt, [pil_image])
            print("Image token added to prompt and image object included.")
        else:
            # Keep as string if no image
             user_input = prompt


        conv.append_message(conv.roles[0], user_input) # Pass the tuple or string
        conv.append_message(conv.roles[1], None)

        print("Processing conversation and image (if any)...")
        # convert_conversation_to_prompts handles the tuple correctly
        all_conv, last_image = convert_conversation_to_prompts(conv)

        stop_words = conv.stop_str
        print(f"Stop words: {stop_words}")

        print("Starting generation...")
        full_response = ""
        for x in kimi_vl_generate(
                conversations=all_conv,
                model=self.model,
                processor=self.processor, 
                stop_words=stop_words,
                max_length=max_length_tokens,
                temperature=temperature,
                top_p=top_p,
            ):
                full_response += x
                chunk = strip_stop_words(full_response, stop_words)
                if len(chunk) > len(strip_stop_words(full_response[:-len(x)], stop_words)): 
                     yield chunk[len(strip_stop_words(full_response[:-len(x)], stop_words)):] 

        print("Generation complete.")
        final_cleaned_response = strip_stop_words(full_response, stop_words)
        last_yielded_part = strip_stop_words(full_response[:-len(x)], stop_words) if 'x' in locals() else ""

        if len(final_cleaned_response) > len(last_yielded_part):
             yield final_cleaned_response[len(last_yielded_part):]


        print(
            f"Finished prediction. Params: temp={temperature}, top_p={top_p}, max_len={max_length_tokens}"
        )