# Genraytor - AI Image Generation Tool

Genraytor is a lightweight command-line tool that generates images from text prompts using Stable Diffusion.

## Features

- Generate images from text prompts
- Random prompt generation
- Customizable image resolution
- Simple, dependency-minimal interface

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ disk space for model weights (automatically downloaded on first run)
- Internet connection (required for first-time setup to download the model)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/abildma/genraytor.git
   cd genraytor
   ```

2. Set up a virtual environment (recommended):
   ```bash
   # Create the virtual environment
   python -m venv venv
   
   # Activate it
   # On Linux/macOS:
   source venv/bin/activate #or source venv/bin/activate.fish
   # On Windows:
   # venv\Scripts\activate
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   
   > **Note:** The first run will download the Stable Diffusion model (4-5GB) automatically to `~/.cache/huggingface/hub/`

## Usage

### Basic Usage

Generate an image from a text prompt:
```bash
python genraytor.py "a majestic lion in the savanna"
```

### Options

- `--random`: Generate an image using a random prompt
- `--res N`: Set image resolution (default: 768, must be multiple of 8)

### Examples

Generate a random image:
```bash
python genraytor.py --random
```

Generate a high-resolution image (1024x1024):
```bash
python genraytor.py --res 1024 "a futuristic cityscape at night"
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors:
1. Try reducing the resolution using `--res` (e.g., `--res 512`)
2. Close other GPU-intensive applications
3. If using a CPU, be patient as generation will be slower

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
