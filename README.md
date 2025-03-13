# GGUF-PyTorch

Goals:
- Run GGUF models in PyTorch (via tensor subclass)
- Run Unsloth Dynamic Quants
- Run it faster than llama.cpp

TODO:
- [ ] Support K-quant

Supported GGML types:
- Q8_0, Q4_0, Q4_1
- Q6_K

## Usage

Chat demo via vLLM

```bash
python chat_vllm.py --model Llama-3.2-1B-Instruct-BF16.gguf --tokenizer meta-llama/Llama-3.2-1B-Instruct
```
