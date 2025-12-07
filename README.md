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
