"""Serve ANY Hugging Face model (your fine-tune included) as an OpenAI-compatible API on Modal.

UNTESTED TEMPLATE - Modal and vLLM change often. Compare with Modal's current
"vLLM inference" example in their docs if something fails.

    pip install modal && modal setup          # one-time login
    modal serve model_service/modal_vllm.py   # temporary dev URL (hot reload)
    modal deploy model_service/modal_vllm.py  # stable URL for the demo

Then in .env:  PROVIDER=local
               LOCAL_BASE_URL=https://<workspace>--junction-llm-serve.modal.run/v1
               LOCAL_MODEL=<same MODEL below>
"""
import subprocess

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"  # swap for your own HF repo, e.g. "yourname/my-finetune"
GPU = "A10G"                         # 24 GB. Use "L40S" or "A100" for bigger models.

image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm", "huggingface_hub")
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)  # avoids re-downloading weights

app = modal.App("junction-llm")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": cache},
    scaledown_window=600,           # keep warm 10 min after last request
    timeout=60 * 60,
    # secrets=[modal.Secret.from_name("huggingface")],  # needed only for private/gated models
)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    subprocess.Popen(
        f"vllm serve {MODEL} --host 0.0.0.0 --port 8000 --max-model-len 8192",
        shell=True,
    )
