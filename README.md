# minillm-lab
Project to learn about llm quantization

### Experiment 1 — Baseline (Transformers, FP16, CPU)

- Model: Qwen/Qwen2.5-0.5B-Instruct
- Runtime: transformers + PyTorch (CPU)
- Machine: Mac, 8 GB RAM, Apple Silicon (arm64)

**Metrics:**
- Model load time: 35.44 s
- RAM after first generation: ~1995 MB
- Prompt: UPSC inflation question (80 new tokens)
- Generation time: 8.46 s
- Throughput: 9.45 tokens/sec
- Subjective: usable but a bit slow; high memory footprint for 8GB machine.

### Download model command:

```bash
huggingface-cli download \               
  Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  "qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  --local-dir models/gguf \
  --local-dir-use-symlinks False
```
### Comparison

| Metric             | FP16 (Transformers CPU)          | GGUF Q4_K_M (llama.cpp + Metal) |
| ------------------ | -------------------------------- | ------------------------------- |
| Model size         | ~1.1 GB                          | ~350 MB                         |
| RAM after load     | ~139 MB → ~2 GB after generation | ~495 MB total                   |
| Load time          | ~35 sec                          | ~17 sec                         |
| Time for 80 tokens | 8.46 sec                         | 1.47 sec                        |
| Throughput         | ~9.45 tok/sec                    | ~54.56 tok/sec                  |
| Backend            | CPU only                         | GPU (Metal)                     |
| Quality            | Good                             | Lower (hallucinated table)      |
