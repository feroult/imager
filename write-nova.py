import argparse
import base64
import json
import os
import subprocess
import boto3
from PIL import Image
from botocore.config import Config
from botocore.exceptions import ClientError

def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        return base64.b64encode(image_bytes).decode('utf-8')

def analyze_text_placement(image_path, text, prompt=None):
    """Use AWS Nova model to determine optimal text placement on an image"""
    # Create Bedrock client - boto3 automatically detects AWS_BEARER_TOKEN_BEDROCK
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
        config=Config(read_timeout=300)
    )
    
    model_id = 'amazon.nova-pro-v1:0'
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Get image format and dimensions
    with Image.open(image_path) as img:
        image_format = img.format.lower()
        width, height = img.size
    
    # Build the analysis prompt
    analysis_prompt = f"""Analyze this image to determine the optimal placement for the text: "{text}"

Image dimensions: {width}x{height} pixels

Please consider:
1. Empty or uncluttered areas where text would be readable
2. Visual balance and composition
3. Avoiding important subjects or focal points
4. Background contrast for text visibility
5. Natural flow and alignment with image elements

Return ONLY a JSON object with these exact fields:
{{
    "x": <x coordinate in pixels for text placement>,
    "y": <y coordinate in pixels for text placement>,
    "font_size": <recommended font size in pixels>,
    "angle": <rotation angle in degrees, 0 for horizontal, negative for counter-clockwise>,
    "reasoning": "<brief explanation of why this placement was chosen>",
    "alternative": {{
        "x": <alternative x coordinate>,
        "y": <alternative y coordinate>,
        "font_size": <alternative font size>,
        "angle": <alternative angle>
    }}
}}

The coordinates should represent where the text baseline starts (bottom-left of the text).
Font size should be proportional to the image (typically 3-8% of image width).
Angle should typically be between -30 and 30 degrees, use 0 for most cases."""

    # Add custom prompt if provided
    if prompt:
        analysis_prompt += f"\n\nAdditional considerations: {prompt}"
    
    analysis_prompt += "\n\nRespond with ONLY the JSON object, no additional text."
    
    body = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": analysis_prompt
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
            "max_new_tokens": 500,
            "temperature": 0.3  # Lower temperature for more consistent JSON output
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
        response_text = response_body['output']['message']['content'][0]['text']
        
        # Parse the JSON response
        try:
            # Clean the response text - remove any markdown code blocks if present
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            placement_data = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['x', 'y', 'font_size', 'angle']
            for field in required_fields:
                if field not in placement_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure numeric types
            placement_data['x'] = int(placement_data['x'])
            placement_data['y'] = int(placement_data['y'])
            placement_data['font_size'] = int(placement_data['font_size'])
            placement_data['angle'] = float(placement_data['angle'])
            
            # Process alternative if present
            if 'alternative' in placement_data:
                alt = placement_data['alternative']
                alt['x'] = int(alt.get('x', placement_data['x']))
                alt['y'] = int(alt.get('y', placement_data['y']))
                alt['font_size'] = int(alt.get('font_size', placement_data['font_size']))
                alt['angle'] = float(alt.get('angle', placement_data['angle']))
            
            return placement_data
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {response_text}")
            raise
        
    except ClientError as e:
        print(f"Error analyzing image: {e}")
        raise

def apply_text_to_image(image_path, text, placement_data, output_path):
    """Apply text to image using ImageMagick convert command"""
    x = placement_data['x']
    y = placement_data['y']
    font_size = placement_data['font_size']
    angle = placement_data['angle']
    
    # Build ImageMagick convert command
    # Note: ImageMagick uses +x+y for position, and rotation is part of annotate
    cmd = [
        'convert',
        image_path,
        '-pointsize', str(font_size),
        '-fill', 'white',  # Default to white, could be made configurable
        '-stroke', 'black',  # Add black stroke for better readability
        '-strokewidth', '1',
        '-annotate', f'+{x}+{y}+{angle}' if angle != 0 else f'+{x}+{y}',
        text,
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error applying text to image: {e}")
        print(f"stderr: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Analyze and apply optimal text placement on an image using AWS Nova and ImageMagick.")
    parser.add_argument('-i', '--image', type=str, required=True, help='Path to the input image file')
    parser.add_argument('-t', '--text', type=str, required=True, help='Text to be placed on the image')
    parser.add_argument('-p', '--prompt', type=str, default=None, 
                       help='Additional prompt for customizing text placement analysis')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output image file path')
    parser.add_argument('--use-alternative', action='store_true',
                       help='Use the alternative text placement instead of primary')
    parser.add_argument('--color', type=str, default='white',
                       help='Text color (default: white)')
    parser.add_argument('--stroke-color', type=str, default='black',
                       help='Text stroke color for outline (default: black)')
    parser.add_argument('--stroke-width', type=str, default='1',
                       help='Text stroke width (default: 1)')
    parser.add_argument('--font', type=str, default=None,
                       help='Font name to use (e.g., Arial-Bold, Helvetica)')
    args = parser.parse_args()

    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        return

    try:
        print(f"Analyzing image for optimal text placement...")
        placement_data = analyze_text_placement(args.image, args.text, args.prompt)
        
        # Print placement data to screen
        print("\n" + "="*50)
        print("Text Placement Analysis:")
        print("="*50)
        print(f"Text: \"{args.text}\"")
        print(f"\nPrimary placement:")
        print(f"  Position: ({placement_data['x']}, {placement_data['y']})")
        print(f"  Font size: {placement_data['font_size']}px")
        print(f"  Angle: {placement_data['angle']}°")
        if 'reasoning' in placement_data:
            print(f"  Reasoning: {placement_data['reasoning']}")
        
        if 'alternative' in placement_data:
            alt = placement_data['alternative']
            print(f"\nAlternative placement:")
            print(f"  Position: ({alt['x']}, {alt['y']})")
            print(f"  Font size: {alt['font_size']}px")
            print(f"  Angle: {alt['angle']}°")
        
        # Use alternative placement if requested
        if args.use_alternative and 'alternative' in placement_data:
            placement_to_use = placement_data['alternative']
            print("\nUsing alternative placement")
        else:
            placement_to_use = placement_data
            print("\nUsing primary placement")
        
        print("="*50)
        print(f"\nApplying text to image...")
        
        # Build ImageMagick command with custom options
        x = placement_to_use['x']
        y = placement_to_use['y']
        font_size = placement_to_use['font_size']
        angle = placement_to_use['angle']
        
        # Build ImageMagick convert command
        cmd = [
            'convert',
            args.image,
            '-pointsize', str(font_size),
            '-fill', args.color,
            '-stroke', args.stroke_color,
            '-strokewidth', args.stroke_width
        ]
        
        # Add font if specified
        if args.font:
            cmd.extend(['-font', args.font])
        
        # Add annotation with position and angle
        if angle != 0:
            cmd.extend(['-annotate', f'+{x}+{y}+{angle}'])
        else:
            cmd.extend(['-annotate', f'+{x}+{y}'])
        
        cmd.append(args.text)
        cmd.append(args.output)
        
        # Execute ImageMagick command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Image with text saved to: {args.output}")
        else:
            print(f"Error applying text to image")
            if result.stderr:
                print(f"Error details: {result.stderr}")
            return
            
    except Exception as e:
        print(f"Failed to process image: {e}")

if __name__ == "__main__":
    main()