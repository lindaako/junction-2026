"""QLoRA fine-tune sized for a 12 GB RTX 3060 (7-8B model, 4-bit).

UNTESTED TEMPLATE - the Unsloth / TRL APIs shift between versions. Do a dry run
on 50 examples tonight so you know it works BEFORE the data arrives.

Setup (Windows: easiest inside WSL2 with the NVIDIA driver installed):
    pip install unsloth trl datasets

Data: a JSONL file where each line is   {"messages": [{"role": "user", "content": "..."},
                                                       {"role": "assistant", "content": "..."}]}
Run:  python finetune/train_lora.py data/train.jsonl
"""
import sys

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

BASE = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # pre-quantized; swap for Llama 3.1 8B, Gemma, etc.
OUT = "outputs/lora"

model, tokenizer = FastLanguageModel.from_pretrained(BASE, max_seq_length=2048, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

ds = load_dataset("json", data_files=sys.argv[1], split="train")
ds = ds.map(lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)})

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=10,
        output_dir=OUT,
        max_seq_length=2048,
    ),
)
trainer.train()

model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
print(f"Saved adapter to {OUT}.")
print("Next: model.push_to_hub_merged('you/my-finetune', tokenizer, save_method='merged_16bit') then point modal_vllm.py at it.")
