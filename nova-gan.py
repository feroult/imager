import argparse
import json
import os
import base64
import textwrap
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from PIL import Image
from botocore.config import Config
from botocore.exceptions import ClientError
from coder.core import Prompter


def light_polish_prompt(prompt):
    """Light prompt polishing - minimal improvements vs heavy enhancement in gen-nova.py"""
    instruction = "Lightly polish this Nova Canvas prompt by fixing grammar and clarity, but keep the original creative intent:\n"
    full_prompt = instruction + prompt
    p = Prompter(textwrap.dedent('''
        You are a professional assistant doing light prompt polishing for AWS Nova Canvas.
        Your task is to:
        - Fix grammar and spelling
        - Clarify ambiguous terms
        - Add minimal technical details if needed
        - Keep the original creative intent completely intact
        - Do NOT add excessive descriptive language
        - Keep it under 1024 characters
        
        Return only the polished prompt, nothing else.
    '''), model='flow-openai-gpt-4o', transient=True)
    polished = p.user(full_prompt)
    print(f"Lightly polished prompt: {polished}")
    return polished.strip()[:1024]


def generate_images_nova_canvas(prompt, num_images=4):
    """Generate multiple images using Nova Canvas"""
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
        config=Config(read_timeout=300)
    )
    
    model_id = 'amazon.nova-canvas-v1:0'
    
    body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": num_images,
            "height": 1024,
            "width": 1024,
            "cfgScale": 8.0,
            "seed": 0
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
        images = response_body.get("images", [])
        
        # Decode base64 images
        decoded_images = []
        for i, base64_image in enumerate(images):
            image_bytes = base64.b64decode(base64_image.encode('ascii'))
            decoded_images.append(image_bytes)
        
        return decoded_images
        
    except ClientError as e:
        print(f"Error generating images: {e}")
        raise


def generate_image_variation(base_image_path, num_images=4):
    """Generate variations of an existing image using Nova Canvas"""
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
        config=Config(read_timeout=300)
    )
    
    model_id = 'amazon.nova-canvas-v1:0'
    
    # Encode base image to base64
    with open(base_image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    body = json.dumps({
        "taskType": "IMAGE_VARIATION", 
        "imageVariationParams": {
            "images": [base64_image],
            "text": ""  # Optional text for variation guidance
        },
        "imageGenerationConfig": {
            "numberOfImages": num_images,
            "quality": "standard",
            "cfgScale": 8.0,
            "seed": 0
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
        images = response_body.get("images", [])
        
        # Decode base64 images
        decoded_images = []
        for base64_image in images:
            image_bytes = base64.b64decode(base64_image.encode('ascii'))
            decoded_images.append(image_bytes)
        
        return decoded_images
        
    except ClientError as e:
        print(f"Error generating image variations: {e}")
        raise


def evaluate_image(image_path, original_prompt):
    """Evaluate how well an image matches the original prompt using Nova Pro"""
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
        config=Config(read_timeout=300)
    )
    
    model_id = 'amazon.nova-pro-v1:0'
    
    # Encode image to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Get image format
    with Image.open(image_path) as img:
        image_format = img.format.lower()
    
    evaluation_prompt = f"""Analyze this image against the target description: '{original_prompt}'. 

Rate how well it matches on a scale of 0.0 to 1.0 considering:
- Subject accuracy (40%)
- Visual quality (30%) 
- Composition (20%)
- Style adherence (10%)

Return ONLY the numeric score (e.g., 0.85), nothing else."""
    
    body = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": evaluation_prompt
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
            "max_new_tokens": 50,
            "temperature": 0.1
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
        evaluation_text = response_body['output']['message']['content'][0]['text'].strip()
        
        # Extract numeric score
        try:
            score = float(evaluation_text.split()[0])
            return max(0.0, min(1.0, score))  # Clamp between 0.0 and 1.0
        except (ValueError, IndexError):
            print(f"Warning: Could not parse score from: {evaluation_text}")
            return 0.0
            
    except ClientError as e:
        print(f"Error evaluating image: {e}")
        return 0.0


def analyze_image(image_path, original_prompt):
    """Get detailed analysis of image for improvement strategy"""
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
        config=Config(read_timeout=300)
    )
    
    model_id = 'amazon.nova-pro-v1:0'
    
    # Encode image to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    with Image.open(image_path) as img:
        image_format = img.format.lower()
    
    analysis_prompt = f"""Analyze this image against target: '{original_prompt}'.

Identify specific issues:
1. Missing elements: [list what's missing]
2. Incorrect elements: [list what's wrong]  
3. Areas needing improvement: [list specific areas]
4. Recommended strategy: [IMAGE_VARIATION for general improvements, or describe specific fixes needed]

Be concise and actionable for image generation improvement."""
    
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
            "temperature": 0.3
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
        analysis = response_body['output']['message']['content'][0]['text']
        return analysis
        
    except ClientError as e:
        print(f"Error analyzing image: {e}")
        return "Analysis failed"


def evaluate_and_analyze_image(image_info):
    """Evaluate and analyze a single image - designed for parallel execution"""
    image_path, original_prompt, candidate_index, prefix = image_info
    
    print(f"Evaluating {prefix}_{candidate_index:02d}...")
    
    try:
        # Evaluate image
        score = evaluate_image(image_path, original_prompt)
        
        # Get analysis for this image
        analysis = analyze_image(image_path, original_prompt)
        
        return {
            'candidate_index': candidate_index,
            'score': score,
            'analysis': analysis,
            'success': True
        }
    except Exception as e:
        print(f"Error evaluating {prefix}_{candidate_index:02d}: {e}")
        return {
            'candidate_index': candidate_index,
            'score': 0.0,
            'analysis': f"Evaluation failed: {e}",
            'success': False
        }


def create_session_structure(output_folder, original_prompt, args):
    """Create initial session structure and config"""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    session_id = f"nova_gan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    config = {
        "session_id": session_id,
        "original_prompt": original_prompt,
        "polished_prompt": None,
        "parameters": {
            "threshold": args.threshold,
            "max_iterations": args.max_iter,
            "polish_prompt": args.polish_prompt,
            "batch_size": 4
        },
        "current_state": {
            "status": "running",
            "current_iteration": 0,
            "best_score": 0.0,
            "best_image": None,
            "threshold_reached": False,
            "start_time": datetime.now().isoformat()
        },
        "iteration_history": []
    }
    
    # Save initial config
    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save original prompt
    with open(output_path / "original_prompt.txt", 'w') as f:
        f.write(original_prompt)
    
    return config


def load_session_config(output_folder):
    """Load existing session configuration"""
    config_path = Path(output_folder) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No session found in {output_folder}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def save_session_config(output_folder, config):
    """Save session configuration"""
    config_path = Path(output_folder) / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def run_iteration(output_folder, config, iteration_num, strategy="TEXT_IMAGE", base_image=None, workers=4):
    """Run a single iteration of generation and evaluation"""
    output_path = Path(output_folder)
    iter_path = output_path / f"iteration_{iteration_num:03d}"
    iter_path.mkdir(exist_ok=True)
    
    print(f"\n--- Iteration {iteration_num} ({strategy}) ---")
    
    # Generate images
    prompt = config.get("polished_prompt") or config["original_prompt"]
    
    if strategy == "TEXT_IMAGE":
        print("Generating 4 new images...")
        images = generate_images_nova_canvas(prompt, 4)
        prefix = "candidate"
    elif strategy == "IMAGE_VARIATION":
        print(f"Generating 4 variations of {base_image}...")
        images = generate_image_variation(base_image, 4)
        prefix = "variation"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Save images temporarily for parallel evaluation
    temp_paths = []
    for i, image_bytes in enumerate(images, 1):
        temp_path = iter_path / f"{prefix}_{i:02d}_temp.png"
        with open(temp_path, "wb") as f:
            f.write(image_bytes)
        temp_paths.append((temp_path, config["original_prompt"], i, prefix))
    
    # Parallel evaluation and analysis
    print("Evaluating images in parallel...")
    candidates = []
    best_score = 0.0
    best_candidate = None
    best_image_path = None
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all evaluation tasks
        future_to_info = {
            executor.submit(evaluate_and_analyze_image, image_info): image_info 
            for image_info in temp_paths
        }
        
        # Process completed evaluations
        for future in as_completed(future_to_info):
            image_info = future_to_info[future]
            temp_path, original_prompt, candidate_index, prefix = image_info
            
            try:
                result = future.result()
                score = result['score']
                analysis = result['analysis']
                success = result['success']
                
                # Rename with score
                final_name = f"{prefix}_{candidate_index:02d}_score_{score:.2f}.png"
                final_path = iter_path / final_name
                temp_path.rename(final_path)
                
                # Save analysis
                analysis_path = iter_path / f"analysis_{candidate_index:02d}.txt"
                with open(analysis_path, 'w') as f:
                    f.write(f"Score: {score:.2f}\n\n{analysis}")
                
                candidates.append({
                    "filename": final_name,
                    "type": strategy,
                    "score": score,
                    "analysis_file": f"analysis_{candidate_index:02d}.txt"
                })
                
                if score > best_score:
                    best_score = score
                    best_candidate = final_name
                    best_image_path = final_path
                
                print(f"  {final_name} - Score: {score:.2f}")
                
            except Exception as e:
                print(f"Error processing {prefix}_{candidate_index:02d}: {e}")
                # Handle failed evaluation by keeping temp file and giving it 0 score
                final_name = f"{prefix}_{candidate_index:02d}_score_0.00.png"
                final_path = iter_path / final_name
                temp_path.rename(final_path)
                
                candidates.append({
                    "filename": final_name,
                    "type": strategy,
                    "score": 0.0,
                    "analysis_file": f"analysis_{candidate_index:02d}.txt"
                })
                
                # Save error analysis
                analysis_path = iter_path / f"analysis_{candidate_index:02d}.txt"
                with open(analysis_path, 'w') as f:
                    f.write(f"Score: 0.00\n\nEvaluation failed: {e}")
    
    # Sort candidates by index to maintain consistent ordering
    candidates.sort(key=lambda x: int(x["filename"].split("_")[1]))
    
    # Create symlink to best image
    best_link = iter_path / "best_image.png"
    if best_link.exists():
        best_link.unlink()
    best_link.symlink_to(best_candidate)
    
    # Save best analysis
    best_analysis_path = iter_path / "best_analysis.txt"
    best_candidate_num = candidates[[c["filename"] for c in candidates].index(best_candidate)]["analysis_file"].split("_")[1].split(".")[0]
    best_analysis_source = iter_path / f"analysis_{best_candidate_num}.txt"
    with open(best_analysis_source, 'r') as src, open(best_analysis_path, 'w') as dst:
        dst.write(src.read())
    
    # Save iteration metadata
    iteration_info = {
        "iteration": iteration_num,
        "strategy": strategy,
        "timestamp": datetime.now().isoformat(),
        "base_image": str(base_image) if base_image else None,
        "candidates": candidates,
        "best_candidate": best_candidate,
        "best_score": best_score
    }
    
    with open(iter_path / "generation_info.json", 'w') as f:
        json.dump(iteration_info, f, indent=2)
    
    return best_score, best_image_path, iteration_info


def main():
    parser = argparse.ArgumentParser(description="Iterative image generation using AWS Nova Canvas GAN-like approach")
    
    # Main arguments
    parser.add_argument('-p', '--prompt', type=str, help='The text prompt to generate images from')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output folder for session files')
    parser.add_argument('-t', '--threshold', type=float, default=0.8, help='Quality threshold to stop iteration (0.0-1.0)')
    parser.add_argument('--max-iter', type=int, default=5, help='Maximum number of iterations')
    parser.add_argument('--polish-prompt', action='store_true', help='Apply light prompt polishing')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers for evaluation (default: 4)')
    
    # Session management
    parser.add_argument('--continue', dest='continue_session', type=str, help='Continue existing session from folder')
    parser.add_argument('--resume', type=str, help='Resume interrupted session')
    parser.add_argument('--inspect', type=str, help='Inspect existing session status')
    parser.add_argument('--from-iteration', type=int, help='Start continuing from specific iteration')
    parser.add_argument('--new-threshold', type=float, help='New threshold when continuing')
    
    args = parser.parse_args()
    
    # Handle inspection
    if args.inspect:
        try:
            config = load_session_config(args.inspect)
            print(f"\nSession: {config['session_id']}")
            print(f"Status: {config['current_state']['status']}")
            print(f"Current iteration: {config['current_state']['current_iteration']}")
            print(f"Best score: {config['current_state']['best_score']:.3f}")
            print(f"Threshold: {config['parameters']['threshold']}")
            print(f"Threshold reached: {config['current_state']['threshold_reached']}")
            if config['current_state']['best_image']:
                print(f"Best image: {config['current_state']['best_image']}")
            return
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    
    # Handle resume
    if args.resume:
        try:
            config = load_session_config(args.resume)
            if config['current_state']['status'] != 'running':
                print(f"Session is not in 'running' state: {config['current_state']['status']}")
                return
            args.output = args.resume
            args.continue_session = args.resume
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    
    # Handle continue
    if args.continue_session:
        try:
            config = load_session_config(args.continue_session)
            args.output = args.continue_session
            
            # Apply new parameters if provided
            if args.new_threshold:
                config['parameters']['threshold'] = args.new_threshold
            if args.max_iter:
                config['parameters']['max_iterations'] = args.max_iter
            
            start_iteration = args.from_iteration or (config['current_state']['current_iteration'] + 1)
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    else:
        # New session
        if not args.prompt:
            print("Error: --prompt is required for new sessions")
            return
        
        config = create_session_structure(args.output, args.prompt, args)
        start_iteration = 1
    
    # Apply prompt polishing if requested
    if args.polish_prompt and not config.get("polished_prompt"):
        polished = light_polish_prompt(config["original_prompt"])
        config["polished_prompt"] = polished
        
        # Save polished prompt
        with open(Path(args.output) / "polished_prompt.txt", 'w') as f:
            f.write(polished)
    
    print(f"Original prompt: {config['original_prompt']}")
    if config.get("polished_prompt"):
        print(f"Polished prompt: {config['polished_prompt']}")
    print(f"Threshold: {config['parameters']['threshold']}")
    print(f"Max iterations: {config['parameters']['max_iterations']}")
    
    # Main iteration loop
    current_best_score = config['current_state']['best_score']
    current_best_image = config['current_state']['best_image']
    
    for iteration in range(start_iteration, config['parameters']['max_iterations'] + 1):
        config['current_state']['current_iteration'] = iteration
        
        # Determine strategy
        if iteration == 1:
            strategy = "TEXT_IMAGE"
            base_image = None
        else:
            strategy = "IMAGE_VARIATION"
            base_image = Path(args.output) / current_best_image if current_best_image else None
        
        # Run iteration
        try:
            best_score, best_image_path, iteration_info = run_iteration(
                args.output, config, iteration, strategy, base_image, args.workers
            )
            
            # Update config
            config['iteration_history'].append(iteration_info)
            
            if best_score > current_best_score:
                current_best_score = best_score
                current_best_image = str(best_image_path.relative_to(Path(args.output)))
                config['current_state']['best_score'] = current_best_score
                config['current_state']['best_image'] = current_best_image
            
            # Check threshold
            if best_score >= config['parameters']['threshold']:
                print(f"\n🎉 Threshold reached! Score: {best_score:.3f} >= {config['parameters']['threshold']}")
                config['current_state']['threshold_reached'] = True
                config['current_state']['status'] = 'completed'
                
                # Create final result symlink
                final_result = Path(args.output) / "final_result.png"
                if final_result.exists():
                    final_result.unlink()
                final_result.symlink_to(current_best_image)
                
                break
                
        except Exception as e:
            print(f"Error in iteration {iteration}: {e}")
            config['current_state']['status'] = 'failed'
            break
        
        # Save progress
        save_session_config(args.output, config)
    
    else:
        # Completed all iterations without reaching threshold
        print(f"\nCompleted {config['parameters']['max_iterations']} iterations. Best score: {current_best_score:.3f}")
        config['current_state']['status'] = 'completed'
        
        # Create final result symlink
        if current_best_image:
            final_result = Path(args.output) / "final_result.png"
            if final_result.exists():
                final_result.unlink()
            final_result.symlink_to(current_best_image)
    
    # Final save
    config['current_state']['completion_time'] = datetime.now().isoformat()
    save_session_config(args.output, config)
    
    print(f"\nSession complete. Results saved in: {args.output}")
    print(f"Final best score: {current_best_score:.3f}")
    if current_best_image:
        print(f"Best image: {current_best_image}")


if __name__ == "__main__":
    main()