# minillm-lab

Project to learn about LLM quantization.

## Experiment 1 — Baseline (Transformers, FP16, CPU)

* **Model:** Qwen/Qwen2.5-0.5B-Instruct
* **Runtime:** Transformers + PyTorch (CPU)
* **Machine:** Mac, 8 GB RAM, Apple Silicon (arm64)

### Metrics

* **Model load time:** 35.44 s
* **RAM after generation:** ~1995 MB
* **Prompt:** UPSC inflation question (80 new tokens)
* **Generation time:** 8.46 s
* **Throughput:** 9.45 tokens/sec
* **Subjective:** Usable but slow; high memory footprint for an 8GB machine.

---

## Experiment 2 — GGUF Q4_K_M (llama.cpp + Metal)

* **Model:** Qwen2.5-0.5B-Instruct (GGUF)
* **Quantization:** Q4_K_M (4-bit grouped, mixed precision)
* **Runtime:** llama-cpp-python (Metal backend)

### Metrics

* **Model load time:** 17.59 s
* **RAM after generation:** ~495.6 MB
* **Generation time:** 1.47 s
* **Throughput:** 54.56 tokens/sec
* **Quality:** Noticeably more hallucinations (e.g., made-up inflation tables)

### Insights

* Significant performance improvement due to quantization + Metal GPU inference.
* Lower memory footprint.
* Some reduction in factual accuracy.

---

## Experiment 3 — GGUF Q5_K_M (llama.cpp + Metal)

* **Model:** Qwen2.5-0.5B-Instruct (GGUF)
* **Quantization:** Q5_K_M (5-bit grouped, higher quality than Q4)

### Metrics

* **Model load time:** 1.56 s
* **RAM after generation:** ~609 MB
* **Generation time:** 1.17 s
* **Throughput:** 68.39 tokens/sec
* **Quality:** Slightly better than Q4 but still hallucinated.

### Insights

* Q5 improves quality while remaining fast and lightweight.
* Memory usage slightly higher than Q4 but well within limits.

---

## Download Commands

### Download Q4_K_M model

```bash
huggingface-cli download \
  Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  "qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  --local-dir models/gguf \
  --local-dir-use-symlinks False
```

### Download Q5_K_M model

```bash
huggingface-cli download \
  Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  "qwen2.5-0.5b-instruct-q5_k_m.gguf" \
  --local-dir models/gguf \
  --local-dir-use-symlinks False
```

---

## Comparison Table

| Metric             | FP16 (Transformers CPU)          | GGUF Q4_K_M (Metal)    | GGUF Q5_K_M (Metal)     |
| ------------------ | -------------------------------- | ---------------------- | ----------------------- |
| Model size         | ~1.1 GB                          | ~350 MB                | ~420 MB (approx)        |
| RAM after load     | ~139 MB → ~2 GB after generation | ~495 MB total          | ~609 MB total           |
| Load time          | ~35 sec                          | ~17 sec                | ~1.56 sec               |
| Time for 80 tokens | 8.46 sec                         | 1.47 sec               | 1.17 sec                |
| Throughput         | ~9.45 tok/sec                    | ~54.56 tok/sec         | ~68.39 tok/sec          |
| Backend            | CPU only                         | GPU (Metal)            | GPU (Metal)             |
| Quality            | Good                             | Lower (hallucinations) | Slightly better than Q4 |

---

## Key Learnings

* Quantization drastically reduces memory and increases inference speed.
* llama.cpp + Metal is highly optimized for Apple Silicon.
* Q4 is fastest & smallest, Q5 trades a bit more memory for better output quality.
* GGUF format includes tokenizer + metadata internally, simplifying inference.

---
