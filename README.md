# GGUF-PyTorch

Goals:
- Run GGUF models in PyTorch (via tensor subclass)
- Run Unsloth Dynamic Quants
- Run it faster than llama.cpp

TODO:
- [ ] Support Q4_0, Q4_1 (Q4_0 and Q4_1 checkpoints uses Q6_K for embedding)
- [ ] Support K-quant

Supported GGML types:
- Q8_0

## Usage

Chat demo via vLLM

```bash
python chat_vllm.py --model Llama-3.2-1B-Instruct-BF16.gguf --tokenizer meta-llama/Llama-3.2-1B-Instruct
```
