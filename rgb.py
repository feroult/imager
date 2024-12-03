import argparse
from PIL import Image

def convert_rgb_to_rgba(input_path, output_path):
    # Open the image
    image = Image.open(input_path)

    # Convert to RGBA
    converted_image = image.convert("RGBA")

    # Save the converted image
    converted_image.save(output_path)

    print(f"Image successfully converted and saved at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert an RGB image to RGBA format.")
    parser.add_argument('input_path', type=str, help='The path to the input RGB image')
    parser.add_argument('output_path', type=str, help='The path to save the converted RGBA image')
    args = parser.parse_args()

    convert_rgb_to_rgba(args.input_path, args.output_path)

if __name__ == "__main__":
    main()