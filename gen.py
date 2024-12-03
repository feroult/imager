import argparse
import textwrap
from openai import OpenAI
from prompter import Prompter

def improve_prompt(prompt):
    instruction = "Turn the following prompt into an image generation prompt:\n"
    full_prompt = instruction + prompt
    p = Prompter(textwrap.dedent('''
        You are a creative assistant. Your task is to take the user's prompt and turn it into an image generation prompt to be sent to another AI.
        Do not just evolve the text or phrasing of the original prompt, but also follow requests inside the original prompt.
        For instance, if a user wants to create an image to represent a quote, first create a creative metaphor description of the image and then send this image description to the image AI.
        The final output should be a clean prompt ready to be sent to the image AI.
    '''), model='flow-openai-gpt-4o', transient=True)
    new_prompt = p.user(full_prompt)
    print(f"Improved prompt: {new_prompt}")
    return new_prompt.strip()

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