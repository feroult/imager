import argparse
import textwrap
from openai import OpenAI
from prompter import Prompter

def improve_prompt(prompt):
    instruction = "Please improve the following prompt to make it more creative and outstanding:\n"
    full_prompt = instruction + prompt
    p = Prompter(textwrap.dedent('''
        You are a creative assistant.
    '''), model='flow-openai-gpt-4o', transient=True)
    new_prompt = p.user(full_prompt)
    return new_prompt

def generate_image(prompt):
    client = OpenAI()
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1792x1024",
        quality="hd",
        n=1,
    )
    image_url = response.data[0].url
    return image_url

def main():
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt using OpenAI's DALL-E model.")
    parser.add_argument('prompt', type=str, help='The text prompt to generate the image from')
    args = parser.parse_args()

    improved_prompt = improve_prompt(args.prompt)
    image_url = generate_image(improved_prompt)
    print(f"Generated image URL: {image_url}")

if __name__ == "__main__":
    main()