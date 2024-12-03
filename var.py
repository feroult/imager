import argparse
from openai import OpenAI

def create_image_variation(image_path):
    client = OpenAI()
    response = client.images.create_variation(
        model="dall-e-2",
        image=open(image_path, "rb"),
        n=1,
        size="1024x1024"
    )
    image_url = response.data[0].url
    return image_url

def main():
    parser = argparse.ArgumentParser(description="Create a variation of an image using OpenAI's DALL-E model.")
    parser.add_argument('image_path', type=str, help='The path to the image to create a variation of')
    args = parser.parse_args()

    image_url = create_image_variation(args.image_path)
    print(f"Generated image variation URL: {image_url}")

if __name__ == "__main__":
    main()