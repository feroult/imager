import argparse
import base64
import json
import os
import boto3
from PIL import Image
from botocore.config import Config
from botocore.exceptions import ClientError

def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        return base64.b64encode(image_bytes).decode('utf-8')

def describe_image(image_path, prompt="Describe this image in detail"):
    """Use AWS Nova model to describe an image"""
    # Create Bedrock client - boto3 automatically detects AWS_BEARER_TOKEN_BEDROCK
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
        config=Config(read_timeout=300)
    )
    
    model_id = 'amazon.nova-pro-v1:0'
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Get image format
    with Image.open(image_path) as img:
        image_format = img.format.lower()
    
    body = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    },
                    {
                        "image": {
                            "format": image_format,
                            "source": {
                                "bytes": base64_image
                            }
                        }
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "max_new_tokens": 1000,
            "temperature": 0.7
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
        description = response_body['output']['message']['content'][0]['text']
        return description
        
    except ClientError as e:
        print(f"Error describing image: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Describe an image using AWS Nova multimodal model.")
    parser.add_argument('-i', '--image', type=str, required=True, help='Path to the image file to describe')
    parser.add_argument('-p', '--prompt', type=str, default="Describe this image in detail", 
                       help='Custom prompt for image description (default: "Describe this image in detail")')
    args = parser.parse_args()

    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        return

    try:
        description = describe_image(args.image, args.prompt)
        print("\nImage Description:")
        print("-" * 50)
        print(description)
        print("-" * 50)
    except Exception as e:
        print(f"Failed to describe image: {e}")

if __name__ == "__main__":
    main()