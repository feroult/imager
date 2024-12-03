import argparse
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

def edit_image(image_path, prompt):
    client = OpenAI()
    response = client.images.edit(
        model="dall-e-2",
        image=open(image_path, "rb"),
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response.data[0].url
    return image_url

def main():
    parser = argparse.ArgumentParser(description="Edit an image based on a text prompt using OpenAI's DALL-E model.")
    parser.add_argument('image_path', type=str, help='The path to the image to be edited')
    parser.add_argument('prompt', type=str, help='The text prompt to edit the image')
    args = parser.parse_args()

    improved_prompt = improve_prompt(args.prompt)
    image_url = edit_image(args.image_path, improved_prompt)
    print(f"Edited image URL: {image_url}")

if __name__ == "__main__":
    main()