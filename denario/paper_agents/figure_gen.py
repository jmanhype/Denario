import re
import json
import base64
import requests
from pathlib import Path
from langchain_core.runnables import RunnableConfig

from .parameters import GraphState
from .tools import LLM_call, json_parser3
from .prompts import figure_planning_prompt, matplotlib_code_prompt
from ..config import INPUT_FILES


def call_nano_banana(api_key: str, prompt: str, model: str = "imagen-3.0-generate-002") -> bytes | None:
    """Call the Google Generative Language API (Nano Banana Pro) to generate an image."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:predict"
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1}
    }

    try:
        resp = requests.post(url, headers=headers, params=params, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        predictions = data.get("predictions", [])
        if predictions and "bytesBase64Encoded" in predictions[0]:
            return base64.b64decode(predictions[0]["bytesBase64Encoded"])
    except Exception as e:
        print(f"Nano Banana API call failed: {e}")

    return None


def extract_code_block(text: str) -> str:
    """Extract a Python code block from LLM output."""
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def generate_matplotlib_figures(state: GraphState) -> GraphState:
    """Ask LLM to write matplotlib code for statistical figures, then exec it."""
    if not state['idea'].get('Results'):
        return state

    PROMPT = matplotlib_code_prompt(state)
    state, code_text = LLM_call(PROMPT, state)
    code = extract_code_block(code_text)

    plots_dir = Path(f"{state['files']['Folder']}/{INPUT_FILES}/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Execute in sandboxed namespace with limited builtins
    namespace = {"__builtins__": __builtins__, "output_dir": str(plots_dir)}
    try:
        exec(code, namespace)
        print(f"  Matplotlib figures generated in {plots_dir}")
    except Exception as e:
        print(f"  Matplotlib generation failed: {e}")

    return state


def generate_figures_node(state: GraphState, config: RunnableConfig):
    """Generate scientific figures using Nano Banana Pro (Google Gemini Image API) and matplotlib."""

    api_key = state["keys"].NANO_BANANA_API_KEY
    plots_dir = Path(f"{state['files']['Folder']}/{INPUT_FILES}/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Matplotlib statistical figures (always attempt if we have Results) ---
    if state['idea'].get('Results'):
        print("Generating matplotlib figures...", end="", flush=True)
        try:
            state = generate_matplotlib_figures(state)
        except Exception as e:
            print(f" matplotlib failed: {e}")

    # --- Phase 2: Nano Banana Pro AI-generated figures (only if API key available) ---
    if api_key:
        print("Generating AI figures via Nano Banana...", end="", flush=True)

        try:
            # Ask LLM what figures the paper needs
            PROMPT = figure_planning_prompt(state)
            state, result = LLM_call(PROMPT, state)
            figure_specs = json_parser3(result)

            figures = figure_specs.get("figures", [])
            generated = 0
            for fig in figures[:6]:  # Max 6 generated figures
                prompt = fig.get("prompt", fig.get("description", ""))
                filename = fig.get("filename", f"ai_figure_{generated}")
                if not prompt:
                    continue

                img_data = call_nano_banana(api_key, prompt)
                if img_data:
                    path = plots_dir / f"{filename}.jpg"
                    with open(path, 'wb') as f:
                        f.write(img_data)
                    generated += 1
                    print(f" [{filename}]", end="", flush=True)

            print(f" {generated} AI figures generated")
        except Exception as e:
            print(f" AI figure generation failed: {e}")
    else:
        print("No GOOGLE_API_KEY set, skipping Nano Banana figure generation")

    # Update plot count
    files = [f for f in plots_dir.iterdir() if f.is_file() and f.name != '.DS_Store']
    state['files']['num_plots'] = len(files)

    return {**state,
            "files": state['files'],
            "tokens": state['tokens']}
