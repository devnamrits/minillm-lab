import os
import time

import psutil
from llama_cpp import Llama

MODEL_PATH = "models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"

def print_mem(label: str):
    process = psutil.Process(os.getpid())
    mem_in_mb = process.memory_info().rss / (1024 ** 2)
    print(f"[MEM] {label}: {mem_in_mb:.1f} MB")

def load_model():
    print("Loading GGUF Q4 model with llama-cpp.....")
    t0 = time.time()

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096, # max context tokens
        n_threads=0, # 0 = auto ( use all CPU cores )
        logits_all=False,
        embedding=False
    )

    t1 = time.time()

    print(f"Model loaded in {t1 - t0: .2f} seconds.")
    print("after GGUF model load")
    return llm

def generate_once(llm: Llama, prompt: str, max_new_tokens: int = 80):
    print("\nRunning GGUF Q4 generation......")
    t0 = time.time()

    result = llm(
        prompt,
        max_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        stop=["</s>", "<|im_start|>", "<|im_end|>"]
    )

    t1 = time.time()

    text = result["choices"][0]["text"]

    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens", None)

    if completion_tokens is None:
        completion_tokens = len(text.split())

    elapsed = t1 - t0
    tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0.0

    print_mem("after GGUF generation")
    print(f"Time for generation: {elapsed:.2f} s")
    print(f"Completion tokens (approx): {completion_tokens}")
    print(f"Throughput: {tok_per_sec:.2f} tokens/sec\n")

    print("========PROMPT========")
    print(prompt)
    print("\n========OUTPUT========")
    print(text)

def main():
    print_mem("startup")
    llm = load_model()

    prompt = (
        "You are an expert UPSC mentor. In 4–5 sentences, explain what inflation is "
        "and how it has affected the Indian economy since 2010, in a way that a UPSC "
        "Mains aspirant would appreciate."
    )

    generate_once(llm, prompt, max_new_tokens=80)

if __name__ == "__main__":
    main()

