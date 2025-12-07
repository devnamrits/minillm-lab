import os
import time
from typing import Tuple

import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def print_mem(label: str):
    """Print current RAM usage"""
    process = psutil.Process(os.getpid())
    mem_in_mb = process.memory_info().rss / (1024 ** 2)
    print(f"[MEM] {label}: {mem_in_mb: .1f} MB")


def load_model() -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    print("Loading tokenizer + model...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="cpu")

    t1 = time.time()

    print(f"Model loaded in {t1 - t0: .2f} seconds.")
    print_mem("after model load")
    return tokenizer, model


def generate_once(tokenizer, model, prompt: str, max_new_tokens: int = 64):
    print("Running generation....")
    inputs = tokenizer(prompt, return_tensors="pt")

    t0 = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    t1 = time.time()

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Compute tokens/sec
    input_len = inputs["input_ids"].shape[-1]
    total_len = output_ids.shape[-1]
    new_tokens = total_len - input_len
    elapsed = t1 - t0
    tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0.0

    print_mem("After generation")
    print(f"Time for generation: {elapsed: .2f} s")
    print(f"New tokens generated: {new_tokens}")
    print(f"Throughput: {tok_per_sec: .2f} tokens/sec\n")

    print("=======PROMPT======")
    print(prompt)
    print("\n=======OUTPUT=======")
    print(output_text)


def main():
    print_mem("startup")
    tokenizer, model = load_model()

    prompt = (
        "You are an expert UPSC mentor. In 4–5 sentences, explain what inflation is "
        "and how it has affected the Indian economy since 2010, in a way that a UPSC "
        "Mains aspirant would appreciate."
    )

    generate_once(tokenizer, model, prompt, max_new_tokens=80)


if __name__ == "__main__":
    main()
