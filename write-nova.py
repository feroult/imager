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

def estimate_text_width(text, font_size):
    """Estimate text width in pixels based on font size and character count"""
    # Rough approximation: average character width is about 0.45 * font_size
    # This varies by font, but gives a reasonable estimate for centering
    avg_char_width = font_size * 0.45
    return int(len(text) * avg_char_width)

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
    analysis_prompt = f"""Analyze this image to determine the optimal placement and styling for the text: "{text}"

Image dimensions: {width}x{height} pixels

Please analyze the image and provide complete text styling that will make the text both readable and aesthetically pleasing. Consider:
1. Empty or uncluttered areas where text would be readable
2. Visual balance and composition
3. Avoiding important subjects or focal points
4. Background colors and contrast for determining text color
5. Image style (modern, vintage, professional, casual) for font selection
6. Natural flow and alignment with image elements
7. **Text inclination/rotation**: Evaluate whether the image has diagonal elements, dynamic composition, or creative context that would benefit from angled text. Use 0° for clean/professional/horizontal layouts, but consider angles when the image has diagonal lines, architectural elements, or needs dynamic visual interest.

Return ONLY a JSON object with these exact fields:
{{
    "x_percent": <x position as percentage of image width (0-100)>,
    "y_percent": <y position as percentage of image height (0-100)>,
    "font_size_percent": <font size as percentage of image width (typically 3-8)>,
    "font": "<font name like Arial, Helvetica, Times-New-Roman, Impact, Georgia, Verdana, Courier, Comic-Sans-MS>",
    "color": "<text color as hex #RRGGBB or color name like white, black, red, blue, yellow>",
    "stroke_color": "<outline color for contrast, use 'none' if no stroke needed>",
    "stroke_width_percent": <stroke width as percentage of font size (typically 5-15 if needed)>,
    "angle": <rotation angle in degrees - consider creative angles, not just 0>,
    "opacity": <text opacity 0-100, where 100 is fully opaque>,
    "reasoning": "<brief explanation of placement and styling choices>",
    "angle_reasoning": "<specific explanation for the angle choice - why 0° or why angled>",
    "alternative": {{
        "x_percent": <alternative x percentage>,
        "y_percent": <alternative y percentage>,
        "font_size_percent": <alternative font size percentage>,
        "font": "<alternative font>",
        "color": "<alternative text color>",
        "stroke_color": "<alternative stroke color>",
        "stroke_width_percent": <alternative stroke width percentage>,
        "angle": <alternative angle - try different from primary>,
        "opacity": <alternative opacity>,
        "angle_reasoning": "<explanation for alternative angle choice>"
    }}
}}

Guidelines:
- x_percent: 0 = left edge, 50 = center, 100 = right edge
- y_percent: Position where you want the CENTER of the text vertically (0 = top, 50 = middle, 100 = bottom)
  Note: The system will automatically adjust for text baseline positioning
- font_size_percent: Typically 3-8% of image width for readability
- stroke_width_percent: 5-15% of font size if stroke needed
- Font: Choose based on image style and mood
- Color: High contrast with background (e.g., white on dark, black on light)
- Angle: Text rotation in degrees. **Choose based on image content:**
  * **0° = horizontal** (DEFAULT for landscapes, portraits, clean/professional images, formal content)
  * **-15 to -5°** = slight counter-clockwise (when image has subtle diagonal elements)
  * **5 to 15°** = slight clockwise (for dynamic energy, following upward slopes)
  * **-30 to -20° or 20 to 30°** = strong angles (only for dramatic compositions, strong diagonal elements)
  * **Key rule**: Use 0° unless the image specifically has diagonal lines, tilted elements, or creative/artistic context
  * Match angles to existing image geometry, don't add angles to static/horizontal compositions
- Opacity: Usually 100, but can be 70-90 for subtle watermarks"""

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
            required_fields = ['x_percent', 'y_percent', 'font_size_percent', 'angle']
            for field in required_fields:
                if field not in placement_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure numeric types for percentages
            placement_data['x_percent'] = float(placement_data['x_percent'])
            placement_data['y_percent'] = float(placement_data['y_percent'])
            placement_data['font_size_percent'] = float(placement_data['font_size_percent'])
            placement_data['angle'] = float(placement_data['angle'])
            placement_data['stroke_width_percent'] = float(placement_data.get('stroke_width_percent', 10))
            placement_data['opacity'] = int(placement_data.get('opacity', 100))
            
            # Convert percentages to pixels
            placement_data['font_size'] = int((placement_data['font_size_percent'] / 100.0) * width)
            placement_data['stroke_width'] = int((placement_data['stroke_width_percent'] / 100.0) * placement_data['font_size'])
            
            # Calculate text width and adjust X position for centering
            text_width = estimate_text_width(text, placement_data['font_size'])
            placement_data['x'] = int((placement_data['x_percent'] / 100.0) * width - text_width / 2)
            
            # Adjust Y position to account for text baseline (add ~70% of font_size to center the text)
            text_height_adjustment = int(placement_data['font_size'] * 0.3)  # Move down to center
            placement_data['y'] = int((placement_data['y_percent'] / 100.0) * height + text_height_adjustment)
            
            # Store text width for display
            placement_data['text_width'] = text_width
            
            # Ensure string types
            placement_data['font'] = str(placement_data.get('font', 'Helvetica'))
            placement_data['color'] = str(placement_data.get('color', 'white'))
            placement_data['stroke_color'] = str(placement_data.get('stroke_color', 'black'))
            
            # Process alternative if present
            if 'alternative' in placement_data:
                alt = placement_data['alternative']
                alt['x_percent'] = float(alt.get('x_percent', placement_data['x_percent']))
                alt['y_percent'] = float(alt.get('y_percent', placement_data['y_percent']))
                alt['font_size_percent'] = float(alt.get('font_size_percent', placement_data['font_size_percent']))
                alt['angle'] = float(alt.get('angle', placement_data['angle']))
                alt['stroke_width_percent'] = float(alt.get('stroke_width_percent', placement_data['stroke_width_percent']))
                alt['opacity'] = int(alt.get('opacity', placement_data['opacity']))
                
                # Convert alternative percentages to pixels
                alt['font_size'] = int((alt['font_size_percent'] / 100.0) * width)
                alt['stroke_width'] = int((alt['stroke_width_percent'] / 100.0) * alt['font_size'])
                
                # Calculate text width and adjust X position for centering
                alt_text_width = estimate_text_width(text, alt['font_size'])
                alt['x'] = int((alt['x_percent'] / 100.0) * width - alt_text_width / 2)
                
                # Adjust Y position to account for text baseline
                alt_text_height_adjustment = int(alt['font_size'] * 0.3)
                alt['y'] = int((alt['y_percent'] / 100.0) * height + alt_text_height_adjustment)
                
                # Store text width for display
                alt['text_width'] = alt_text_width
                
                alt['font'] = str(alt.get('font', placement_data['font']))
                alt['color'] = str(alt.get('color', placement_data['color']))
                alt['stroke_color'] = str(alt.get('stroke_color', placement_data['stroke_color']))
            
            return placement_data
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {response_text}")
            raise
        
    except ClientError as e:
        print(f"Error analyzing image: {e}")
        raise

def apply_text_to_image(image_path, text, placement_data, output_path):
    """Apply text to image using ImageMagick convert command with model-provided styling"""
    x = placement_data['x']
    y = placement_data['y']
    font_size = placement_data['font_size']
    angle = placement_data.get('angle', 0)
    font = placement_data.get('font', 'Helvetica')
    color = placement_data.get('color', 'white')
    stroke_color = placement_data.get('stroke_color', 'none')
    stroke_width = placement_data.get('stroke_width', 0)
    opacity = placement_data.get('opacity', 100)
    
    # Build ImageMagick convert command
    cmd = [
        'convert',
        image_path,
        '-font', font,
        '-pointsize', str(font_size),
        '-fill', color
    ]
    
    # Add stroke if specified
    if stroke_color != 'none' and stroke_width > 0:
        cmd.extend(['-stroke', stroke_color, '-strokewidth', str(stroke_width)])
    
    # Add opacity if not 100%
    if opacity < 100:
        cmd.extend(['-fill-opacity', f'{opacity}%'])
    
    # Handle text rotation properly
    if angle != 0:
        # Use draw command for better angle control
        draw_cmd = f"translate {x},{y} rotate {angle} text 0,0 '{text}'"
        cmd.extend(['-draw', draw_cmd])
    else:
        # Standard annotation for non-rotated text
        cmd.extend(['-annotate', f'+{x}+{y}', text])
    
    cmd.append(output_path)
    
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
    parser.add_argument('--override-color', type=str, default=None,
                       help='Override model-suggested text color')
    parser.add_argument('--override-font', type=str, default=None,
                       help='Override model-suggested font')
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
        print(f"  Position: {placement_data['x_percent']:.1f}% x, {placement_data['y_percent']:.1f}% y ({placement_data['x']}, {placement_data['y']} px, centered)")
        print(f"  Text width: ~{placement_data.get('text_width', 0)}px")
        print(f"  Font: {placement_data.get('font', 'Helvetica')}")
        print(f"  Font size: {placement_data['font_size_percent']:.1f}% ({placement_data['font_size']}px)")
        print(f"  Color: {placement_data.get('color', 'white')}")
        print(f"  Stroke: {placement_data.get('stroke_color', 'none')} ({placement_data['stroke_width_percent']:.1f}% = {placement_data.get('stroke_width', 0)}px)")
        print(f"  Angle: {placement_data.get('angle', 0)}°")
        print(f"  Opacity: {placement_data.get('opacity', 100)}%")
        if 'reasoning' in placement_data:
            print(f"  Reasoning: {placement_data['reasoning']}")
        if 'angle_reasoning' in placement_data:
            print(f"  Angle reasoning: {placement_data['angle_reasoning']}")
        
        if 'alternative' in placement_data:
            alt = placement_data['alternative']
            print(f"\nAlternative placement:")
            print(f"  Position: {alt['x_percent']:.1f}% x, {alt['y_percent']:.1f}% y ({alt['x']}, {alt['y']} px, centered)")
            print(f"  Text width: ~{alt.get('text_width', 0)}px")
            print(f"  Font: {alt.get('font', 'Helvetica')}")
            print(f"  Font size: {alt['font_size_percent']:.1f}% ({alt['font_size']}px)")
            print(f"  Color: {alt.get('color', 'white')}")
            print(f"  Stroke: {alt.get('stroke_color', 'none')} ({alt['stroke_width_percent']:.1f}% = {alt.get('stroke_width', 0)}px)")
            print(f"  Angle: {alt.get('angle', 0)}°")
            print(f"  Opacity: {alt.get('opacity', 100)}%")
            if 'angle_reasoning' in alt:
                print(f"  Angle reasoning: {alt['angle_reasoning']}")
        
        # Use alternative placement if requested
        if args.use_alternative and 'alternative' in placement_data:
            placement_to_use = placement_data['alternative']
            print("\nUsing alternative placement")
        else:
            placement_to_use = placement_data
            print("\nUsing primary placement")
        
        # Apply user overrides if specified
        if args.override_color:
            placement_to_use['color'] = args.override_color
            print(f"\nOverriding color with: {args.override_color}")
        if args.override_font:
            placement_to_use['font'] = args.override_font
            print(f"Overriding font with: {args.override_font}")
        
        print("="*50)
        print(f"\nApplying text to image...")
        
        # Use the apply_text_to_image function with model-provided styling
        success = apply_text_to_image(args.image, args.text, placement_to_use, args.output)
        
        if success:
            print(f"✓ Image with text saved to: {args.output}")
        else:
            print(f"Failed to create output image")
            
    except Exception as e:
        print(f"Failed to process image: {e}")

if __name__ == "__main__":
    main()