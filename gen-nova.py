import argparse
import textwrap
import base64
import json
import io
import os
import boto3
from PIL import Image
from botocore.config import Config
from botocore.exceptions import ClientError
from coder.core import Prompter

def improve_prompt(prompt):
    instruction = "Turn the following prompt into a high-quality Nova Canvas image generation prompt:\n"
    full_prompt = instruction + prompt
    p = Prompter(textwrap.dedent('''
        You are a professional assistant optimizing prompts for AWS Nova Canvas image generation.
        Nova Canvas works best with descriptive, caption-like prompts rather than commands.
        Transform the user's prompt into a clear, descriptive prompt that includes:
        - Subject description
        - Environment/setting
        - Optional: lighting, camera position, visual style
        
        Keep it under 1024 characters and avoid negation words like "no" or "without".
        Use descriptive language like "realistic editorial photo of..." or "detailed illustration of..."
        The final output should be a clean, descriptive prompt optimized for Nova Canvas.
    '''), model='flow-openai-gpt-4o', transient=True)
    new_prompt = p.user(full_prompt)
    print(f"Nova Canvas optimized prompt: {new_prompt}")
    return new_prompt.strip()[:1024]

def generate_image(prompt, output_path):
    # Create Bedrock client - boto3 automatically detects AWS_BEARER_TOKEN_BEDROCK
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
        config=Config(read_timeout=300)
    )
    
    model_id = 'amazon.nova-canvas-v1:0'
    
    body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 1024,
            "width": 1024,
            "cfgScale": 8.0,
            "seed": 0
        }
    })
    
    try:
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get("body").read())
        base64_image = response_body.get("images")[0]
        image_bytes = base64.b64decode(base64_image.encode('ascii'))
        
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        print(f"Image successfully generated and saved at {output_path}")
        
    except ClientError as e:
        print(f"Error generating image: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt using AWS Nova Canvas.")
    parser.add_argument('-p', '--prompt', type=str, required=True, help='The text prompt to generate the image from')
    parser.add_argument('-o', '--output', type=str, required=True, help='The path to save the generated image')
    args = parser.parse_args()

    improved_prompt = improve_prompt(args.prompt)
    generate_image(improved_prompt, args.output)

if __name__ == "__main__":
    main()