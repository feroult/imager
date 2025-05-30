import argparse
import textwrap
import base64
from openai import OpenAI
from prompter import Prompter

def improve_prompt(prompt):
    instruction = "Turn the following prompt into a high-quality image generation prompt:\n"
    full_prompt = instruction + prompt
    p = Prompter(textwrap.dedent('''
        You are a professional assistant. Your task is to take the user's prompt and turn it into a high-quality image generation prompt to be sent to another AI.
        Focus on enhancing the clarity, precision, and technical details of the prompt to ensure the best possible image quality.
        Avoid adding unnecessary creative elements or metaphorical descriptions. The final output should be a clean and precise prompt ready to be sent to the image AI.
    '''), model='flow-openai-gpt-4o', transient=True)
    new_prompt = p.user(full_prompt)
    print(f"Improved prompt: {new_prompt}")
    return new_prompt.strip()

def generate_image(prompt, output_path):
    client = OpenAI()
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1536x1024",
        quality="high",
        n=1,
    )
    image_base64 = response.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    with open(output_path, "wb") as f:
        f.write(image_bytes)
    print(f"Image successfully generated and saved at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt using OpenAI's DALL-E model.")
    parser.add_argument('-p', '--prompt', type=str, required=True, help='The text prompt to generate the image from')
    parser.add_argument('-o', '--output', type=str, required=True, help='The path to save the generated image')
    args = parser.parse_args()

    #improved_prompt = improve_prompt(args.prompt)
    improved_prompt = improve_prompt(args.prompt)
    generate_image(improved_prompt, args.output)

if __name__ == "__main__":
    main()