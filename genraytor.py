#!/usr/bin/env python3
"""genraytor – simple CLI to generate an image from a text prompt.

Usage:
    genraytor "A cosy wooden house in winter"

It will generate a PNG with a filename derived from the prompt and a matching
`.txt` containing a short AI-written description.
"""
from __future__ import annotations

import sys
import argparse
import threading
import itertools
import time
from pathlib import Path
from contextlib import contextmanager
import colorama

# Initialize colorama and create color constants
colorama.init()
COLOR_INFO = colorama.Fore.YELLOW
COLOR_SUCCESS = colorama.Fore.GREEN
COLOR_ERROR = colorama.Fore.RED
COLOR_RESET = colorama.Style.RESET_ALL

@contextmanager
def spinner(message: str):
    """Simple spinner context manager with elapsed time."""
    stop = threading.Event()
    
    def run():
        start_time = time.time()
        spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        
        while not stop.is_set():
            elapsed = int(time.time() - start_time)
            sys.stdout.write(f'\r{COLOR_INFO}{next(spinner)} {message} ({elapsed}s){COLOR_RESET} ')
            sys.stdout.flush()
            time.sleep(0.1)
        
        # Clear the line when done
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
    
    t = threading.Thread(target=run)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()

from sd_render import sd_generate, call_ollama, DEFAULT_MODEL


def slugify(text: str) -> str:
    """Make a safe filename slug from the prompt."""
    return (
        text.lower()
        .strip()
        .replace(" ", "_")
        .replace("/", "-")
    ) or "image"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion.")
    parser.add_argument("prompt", nargs="*", help="Text prompt to render")
    parser.add_argument("--random", action="store_true",
                      help="Generate using a randomly constructed prompt")
    parser.add_argument("--res", type=int, default=768,
                      help="Square resolution in pixels (default: 768). 512 for low VRAM, >1024 may cause OOM.")

    args = parser.parse_args()

    # Determine prompt
    if args.random:
        import random
        adjectives = [
            "mystical", "vibrant", "ancient", "dreamy", "futuristic", "serene", "whimsical", "epic",
        ]
        nouns = [
            "forest", "castle", "ocean", "dragon", "cityscape", "nebula", "robot", "garden",
        ]
        prompt = f"{random.choice(adjectives)} {random.choice(nouns)}"
        print(f"[info] Random prompt: '{prompt}'")
    elif args.prompt:
        prompt = " ".join(args.prompt)
    else:
        sys.exit("[error] Provide a PROMPT or use --random.")

    # Set resolution (square)
    width = height = args.res
    outfile = Path(f"{slugify(prompt)}.png")

    # Generate image
    try:
        with spinner("Generating image"):
            sd_generate(prompt, outfile, height=height, width=width)
        print(f"{COLOR_SUCCESS}[ok] Image saved to {outfile}{COLOR_RESET}")
    except Exception as e:
        sys.exit(f"{COLOR_ERROR}[error] Image generation failed: {e}{COLOR_RESET}")


if __name__ == "__main__":
    main()
