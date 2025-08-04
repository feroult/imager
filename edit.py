import argparse
import base64
import textwrap
from openai import OpenAI
from coder.core import Prompter

def improve_prompt(prompt):
    instruction = "Turn the following prompt into a high-quality image editing prompt:\n"
    full_prompt = instruction + prompt
    p = Prompter(textwrap.dedent('''
        You are a professional assistant. Your task is to take the user's prompt and turn it into a high-quality image editing prompt to be sent to another AI.
        Focus on enhancing the clarity, precision, and technical details of the prompt to ensure the best possible image quality.
        Avoid adding unnecessary creative elements or metaphorical descriptions. The final output should be a clean and precise prompt ready to be sent to the image AI.
    '''), model='flow-openai-gpt-4o', transient=True)
    new_prompt = p.user(full_prompt)
    print(f"Improved prompt: {new_prompt}")
    return new_prompt.strip()

def edit_image(input_images, prompt, output_path):
    client = OpenAI()

    # Open all input images
    image_files = [open(image_path, "rb") for image_path in input_images]

    # Call the OpenAI API for image editing
    result = client.images.edit(
        model="gpt-image-1",
        image=image_files,
        # size="1536x1024",
        size="1024x1024",
        # quality="high",        
        quality="medium",
        prompt=prompt
    )

    # Decode the base64 image data
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    # Save the image to the specified output path
    with open(output_path, "wb") as f:
        f.write(image_bytes)

    print(f"Edited image successfully saved at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Edit an image based on a text prompt using OpenAI's image editing model.")
    parser.add_argument('-i', '--input', type=str, nargs='+', required=True, help='Paths to the input images (multiple allowed)')
    parser.add_argument('-p', '--prompt', type=str, required=True, help='The text prompt to edit the image')
    parser.add_argument('-o', '--output', type=str, required=True, help='The path to save the edited image')
    args = parser.parse_args()

    improved_prompt = args.prompt
    # improved_prompt = improve_prompt(args.prompt)
    edit_image(args.input, improved_prompt, args.output)

if __name__ == "__main__":
    main()