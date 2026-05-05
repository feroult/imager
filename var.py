import argparse
import base64
import os
from openai import OpenAI

def create_image_variation(image_path, output_path):
    client = OpenAI()
    with open(image_path, "rb") as f:
        result = client.images.edit(
            model="gpt-image-2",
            image=f,
            prompt="Create a creative variation of this image, keeping the overall style and subject but with meaningful differences in composition, lighting, or details.",
            size="1024x1024",
            quality="medium",
            n=1,
        )
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    with open(output_path, "wb") as out:
        out.write(image_bytes)
    print(f"Image variation saved at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create a variation of an image using OpenAI's gpt-image-2 model.")
    parser.add_argument('image_path', type=str, help='Path to the source image')
    parser.add_argument('-o', '--output', type=str, help='Output path (default: <image_path>_var.png)')
    args = parser.parse_args()

    output_path = args.output or os.path.splitext(args.image_path)[0] + "_var.png"
    create_image_variation(args.image_path, output_path)

if __name__ == "__main__":
    main()
