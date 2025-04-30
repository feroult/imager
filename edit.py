import argparse
import base64
import textwrap
from openai import OpenAI
from prompter import Prompter

def improve_prompt(prompt):
    instruction = "Turn the following prompt into an image editing prompt:\n"
    full_prompt = instruction + prompt
    p = Prompter(textwrap.dedent('''
        You are a creative assistant. Your task is to take the user's prompt and turn it into an image editing prompt to be sent to another AI.
        Do not just evolve the text or phrasing of the original prompt, but also follow requests inside the original prompt.
        For instance, if a user wants to edit an image to represent a quote, first create a creative metaphor description of the image and then send this image description to the image AI.
        The final output should be a clean prompt ready to be sent to the image AI.
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

    improved_prompt = improve_prompt(args.prompt)
    edit_image(args.input, improved_prompt, args.output)

if __name__ == "__main__":
    main()