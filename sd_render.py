#!/usr/bin/env python3
"""ascii_art_cli.py – Generate ASCII art from a text prompt using a local LLM.

Usage examples:
    # Plain interactive
    python ascii_art_cli.py "A cute cat playing guitar"

    # Force figlet fallback (no model call)
    python ascii_art_cli.py --offline "Hello world"

    # Specify an ollama model (default: llama3)
    python ascii_art_cli.py --model ascii-art-llama "Dragon breathing fire"

The script tries to call an ollama model installed locally.  Install ollama:
    curl https://ollama.ai/install.sh | sh
and pull a model, e.g.
    ollama pull llama3

"""
from __future__ import annotations

import argparse
import itertools
import logging
import os
import shutil
import subprocess
import sys
import time
import warnings
from typing import List

# Suppress specific warnings
warnings.filterwarnings("ignore", message="You have disabled the safety checker for")
warnings.filterwarnings("ignore", module="diffusers")

from pyfiglet import Figlet
from pathlib import Path
from diffusers import StableDiffusionPipeline
import torch
from colorama import Fore, Style, init as colorama_init

# Suppress diffusers logging
diffusers_logger = logging.getLogger("diffusers")
diffusers_logger.setLevel(logging.ERROR)

# Initialize colorama for consistent colors across platforms
colorama_init()

OLLAMA_CMD = shutil.which("ollama")
DEFAULT_MODEL = "llama3"
BASE_SYSTEM_PROMPT = (
    "You are an AI artist that creates detailed, high-quality ASCII art. "
    "For each prompt, analyze the scene and generate ASCII art with appropriate characters "
    "to represent different elements (e.g., use '@' for sun, '/' for grass, '#' for mountains). "
    "Consider perspective, shading, and composition. Use multiple lines to create depth. "
    "Return ONLY the ASCII art without any character-per-line restriction. "
    "Include a brief description of the color scheme in the format: "
    "# COLORS: sun=yellow, grass=green, sky=blue, etc."
)


# Cached global pipeline to avoid reloading on every call
_SD_PIPELINE: StableDiffusionPipeline | None = None
_SD_DEVICE: str | None = None

# Custom progress callback class
class ProgressCallback:
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.bar_length = 30
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        # Print initial newline to separate from command output
        print()
    
    def _get_status(self):
        progress = min(1.0, max(0, self.current_step / self.total_steps))
        filled = int(self.bar_length * progress)
        bar = '█' * filled + '░' * (self.bar_length - filled)
        elapsed = time.time() - self.start_time
        
        # Format time as MM:SS
        def format_time(seconds):
            return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
        
        time_elapsed = format_time(elapsed)
        
        # Create status line with color
        spinner = f"{Fore.YELLOW}{next(self.spinner)}{Style.RESET_ALL}"
        bar_colored = f"{Fore.GREEN}{bar[:filled]}{Fore.LIGHTBLACK_EX}{bar[filled:]}{Style.RESET_ALL}"
        
        return (f"{spinner} {bar_colored} {Fore.CYAN}{self.current_step:2d}/{self.total_steps} "
                f"{Fore.WHITE}({progress:.0%}) {Fore.YELLOW}{time_elapsed}{Style.RESET_ALL}")
    
    def __call__(self, step: int, timestep: int, latents: torch.FloatTensor) -> None:
        self.current_step = step + 1
        status = self._get_status()
        # Clear line and print status
        sys.stdout.write(f'\r{status}')
        sys.stdout.flush()
        
        if self.current_step >= self.total_steps:
            # Clear the line when done
            time.sleep(0.5)  # Let the final state be visible briefly
            sys.stdout.write('\r' + ' ' * len(status) + '\r')
            sys.stdout.flush()

def _load_pipeline(device: str) -> StableDiffusionPipeline:
    """(Re)load the global pipeline on the requested device with memory-friendly tweaks."""
    global _SD_PIPELINE, _SD_DEVICE
    if _SD_PIPELINE is not None and _SD_DEVICE == device:
        return _SD_PIPELINE
    
    model_id = "runwayml/stable-diffusion-v1-5"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Suppress all non-error messages
    logging.basicConfig(level=logging.ERROR)
    
    # Clear any existing output
    sys.stdout.write('\r' + ' ' * 120 + '\r')
    sys.stdout.flush()
    
    # Load pipeline with progress bar
    with torch.inference_mode():
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            safety_checker=None,
            use_safetensors=True,
            local_files_only=False,
            ignore_mismatched_sizes=True
        )
        
        # Memory-saving methods
        pipe.enable_attention_slicing()
        
        # Only enable CPU offload if not using CUDA
        if device != "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    
    _SD_PIPELINE, _SD_DEVICE = pipe, device
    return pipe

def sd_generate(
    prompt: str,
    out_path: Path,
    *,
    height: int = 512,
    width: int = 512,
    steps: int = 30,
    guidance: float = 7.5,
) -> None:
    """Generate an image with Stable Diffusion (runwayml/sd-v1-5) and save to out_path."""
    device_preference = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # Initialize progress callback
        progress_callback = ProgressCallback(total_steps=steps)
        
        # Load the pipeline
        pipe = _load_pipeline(device_preference)
        
        # Generate the image with progress updates
        with torch.inference_mode():
            if device_preference == "cuda":
                with torch.autocast("cuda"):
                    image = pipe(
                        prompt,
                        height=height,
                        width=width,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        callback=progress_callback,
                        callback_steps=1
                    ).images[0]
            else:
                image = pipe(
                    prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    callback=progress_callback,
                    callback_steps=1
                ).images[0]
        
        # Save the image
        image.save(out_path)
        
    except Exception as e:
        # Clear any progress output before showing error
        sys.stdout.write('\r' + ' ' * 120 + '\r')
        sys.stdout.flush()
        raise RuntimeError(f"Image generation failed: {str(e)}") from e

# (Deprecated) Ollama image path retained for reference but unused now
def call_ollama_image(prompt:str, model:str, temperature:float, out_path:Path)->None:
    raise RuntimeError("Ollama image models not available; Stable Diffusion path used instead.")
    """Generate an image via ollama and save it to out_path (PNG)."""
    if not OLLAMA_CMD:
        raise RuntimeError("ollama executable not found in PATH.")
    cmd=[
        OLLAMA_CMD,
        "run",
        model,
        "--format","png",
        prompt,
    ]
    env=dict(os.environ)
    if temperature is not None:
        env["OLLAMA_TEMPERATURE"]=str(temperature)
    # capture binary stdout
    result=subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE)
    out_path.write_bytes(result.stdout)

def build_system_prompt(min_lines:int, width:int)->str:
    return (
        BASE_SYSTEM_PROMPT +
        f" Use at least {min_lines} lines of ASCII. Target width is about {width} characters." )

def call_ollama(prompt: str, model: str, temperature: float, num_predict:int|None, min_lines:int, width:int) -> str:
    """Invoke ollama with the given prompt and return stdout (stripped)."""
    if not OLLAMA_CMD:
        raise RuntimeError("ollama executable not found in PATH.")

    system_prompt=build_system_prompt(min_lines,width)
    full_prompt = f"{system_prompt}\n{prompt}"
    env = dict(os.environ)
    if num_predict:
        env["OLLAMA_NUM_PREDICT"] = str(num_predict)
    if temperature is not None:
        # Works on older ollama by setting env var; newer versions ignore if flag also provided
        env["OLLAMA_TEMPERATURE"] = str(temperature)
    cmd: List[str] = [
        OLLAMA_CMD,
        "run",
        model,
        full_prompt,
    ]
    try:
        result = subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ollama failed (exit {exc.returncode})") from exc

    return result.stdout.strip()


def generate_figlet(text: str) -> str:
    fig = Figlet(font="standard", width=120)
    return fig.renderText(text)


def apply_context_coloring(text: str, prompt: str) -> str:
    # Extract color scheme from the ASCII art if present
    color_scheme = {}
    lines = text.splitlines()
    color_line = next((line for line in lines if line.startswith('# COLORS:')), None)
    
    if color_line:
        # Parse color scheme
        scheme_str = color_line.replace('# COLORS:', '').strip()
        for pair in scheme_str.split(', '):
            if '=' in pair:
                element, color = pair.split('=')
                color_scheme[element.strip()] = color.strip()
    
    # Default color mappings
    default_colors = {
        'sun': Fore.YELLOW,
        'sky': Fore.BLUE,
        'grass': Fore.GREEN,
        'mountain': Fore.WHITE,
        'water': Fore.CYAN,
        'tree': Fore.GREEN,
        'cloud': Fore.WHITE,
        'flower': Fore.MAGENTA,
        'path': Fore.YELLOW,
        'road': Fore.YELLOW,
        'building': Fore.WHITE,
        'rock': Fore.LIGHTBLACK_EX,
        'sand': Fore.YELLOW
    }
    
    # Update default colors based on prompt context
    if 'sunset' in prompt.lower():
        default_colors['sun'] = Fore.RED
        default_colors['sky'] = Fore.LIGHTRED_EX
    elif 'dawn' in prompt.lower():
        default_colors['sun'] = Fore.YELLOW
        default_colors['sky'] = Fore.MAGENTA
    
    # Create color map from ASCII characters
    color_map = {
        '@': default_colors.get('sun', Fore.YELLOW),  # Sun
        '#': default_colors.get('mountain', Fore.WHITE),  # Mountains
        '/': default_colors.get('grass', Fore.GREEN),  # Grass
        '~': default_colors.get('water', Fore.CYAN),  # Water
        '^': default_colors.get('tree', Fore.GREEN),  # Trees
        'O': default_colors.get('cloud', Fore.WHITE),  # Clouds
        '-': default_colors.get('path', Fore.YELLOW),  # Paths
        '=': default_colors.get('path', Fore.YELLOW),  # Roads
        '*': default_colors.get('flower', Fore.MAGENTA),  # Flowers
        'o': default_colors.get('rock', Fore.LIGHTBLACK_EX),  # Rocks
        'S': default_colors.get('sand', Fore.YELLOW),  # Sand
        'B': default_colors.get('building', Fore.WHITE),  # Buildings
    }
    
    # Apply colors to the ASCII art
    out_lines = []
    for line in lines:
        if line.startswith('# COLORS:'):  # Skip color scheme line
            continue
        colored = ""
        for char in line:
            if char in color_map:
                colored += color_map[char] + char
            else:
                colored += char
        out_lines.append(colored + Style.RESET_ALL)
    
    return "\n".join(out_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ASCII art from text using a local LLM or figlet fallback.")
    parser.add_argument("prompt", nargs="*", help="Description of the ASCII art")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature (ollama) e.g. 0.2–1.5")
    parser.add_argument("--repeat", type=int, default=1, help="Generate N variations (LLM mode)")
    parser.add_argument("--width", type=int, default=120, help="Target width of ASCII art")
    parser.add_argument("--lines", type=int, default=20, help="Minimum number of ASCII art lines")
    parser.add_argument("--tokens", type=int, default=None, help="Maximum tokens (env OLLAMA_NUM_PREDICT)")
    parser.add_argument("--image", action="store_true", help="Generate a PNG image instead of ASCII art (uses image model)")
    parser.add_argument("--out", type=str, default=None, help="Output PNG filename (image mode)")
    parser.add_argument("--offline", action="store_true", help="Skip LLM and just run figlet")

    args = parser.parse_args()

    colorama_init()

    if not args.prompt:
        prompt = input("Describe the ASCII art you want: ").strip()
    else:
        prompt = " ".join(args.prompt)

    ascii_art: str
    if args.image:
        out_file = args.out or f"{prompt[:32].strip().replace(' ','_')}.png"
        try:
            sd_generate(prompt, Path(out_file))
            print(f"[ok] Image saved to {out_file}")
        except Exception as e:
            print(f"[error] Image generation failed: {e}", file=sys.stderr)
        return

    if not args.offline:
        arts: List[str] = []
        for _ in range(max(1, args.repeat)):
            try:
                art = call_ollama(prompt, args.model, args.temp, args.tokens, args.lines, args.width)
            except Exception as e:
                print(f"[warn] LLM generation failed: {e}. Falling back to figlet.\n", file=sys.stderr)
                art = generate_figlet(prompt)
            arts.append(art)
    else:
        arts = [generate_figlet(prompt)]

    for idx, art in enumerate(arts, 1):
        if args.repeat > 1:
            print(f"\n--- variation {idx}/{len(arts)} ---")
        print(apply_context_coloring(art, prompt))


if __name__ == "__main__":
    main()
