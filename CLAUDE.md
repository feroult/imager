# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an image generation and manipulation toolkit with multiple AI-powered backends. The project provides command-line utilities for:

- **Image Generation**: Using OpenAI DALL-E (gpt-image-1) and AWS Nova Canvas
- **Image Description**: Using AWS Nova Pro multimodal model 
- **Image Editing**: Using OpenAI's image editing capabilities
- **Image Utilities**: RGB to RGBA conversion, image variations

## Core Architecture

### AI Service Integration
The project integrates with multiple AI services:
- **OpenAI**: Uses `gpt-image-1` model for generation and editing, `dall-e-2` for variations
- **AWS Bedrock**: Uses `amazon.nova-canvas-v1:0` for generation and `amazon.nova-pro-v1:0` for multimodal tasks
- **Flow-Coder**: Uses `flow-openai-gpt-4o` for prompt optimization

### Prompt Enhancement System
Most scripts use a two-stage approach:
1. Original user prompt is enhanced by Flow-Coder's GPT-4o
2. Enhanced prompt is sent to the target image generation service

## Command Usage

### Environment Setup
```bash
poetry install
poetry shell  # Optional: activate virtual environment
```

### Script Execution
All scripts should be run with Poetry:
```bash
poetry run python3 <script_name>.py [arguments]
```

Or within activated shell:
```bash
python3 <script_name>.py [arguments]
```

### Available Scripts

**OpenAI Image Generation** (`gen.py`):
```bash
poetry run python3 gen.py -p "your prompt" -o output.png
```

**AWS Nova Canvas Generation** (`gen-nova.py`):
```bash
poetry run python3 gen-nova.py -p "your prompt" -o output.png
```

**Image Description** (`describe-nova.py`):
```bash
poetry run python3 describe-nova.py -i image.jpg [-p "custom prompt"]
```

**Image Editing** (`edit.py`):
```bash
poetry run python3 edit.py -i input1.jpg input2.jpg -p "edit prompt" -o output.png
```

**RGB to RGBA Conversion** (`rgb.py`):
```bash
poetry run python3 rgb.py input.jpg output.png
```

**Image Variations** (`var.py`):
```bash
poetry run python3 var.py image.png
```

## Authentication

- **OpenAI**: Requires `OPENAI_API_KEY` environment variable
- **AWS**: Uses `AWS_BEARER_TOKEN_BEDROCK` (automatically detected by boto3)
- **Flow-Coder**: Authentication handled by flow-coder library

## Key Dependencies

- `openai>=1.93.0`: OpenAI API client
- `boto3>=1.35.0`: AWS services client
- `pillow>=11.3.0`: Image processing
- `flow-coder`: Prompt enhancement (custom wheel)

## Development Notes

- Python 3.12 required (`~3.12` in pyproject.toml)
- Project uses `package-mode = false` (scripts-only project)
- Poetry used for dependency management
- Image outputs are typically 1024x1024 or 1536x1024 depending on service