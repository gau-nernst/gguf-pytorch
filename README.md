# GGUF-PyTorch

Goals:
- Run GGUF models in PyTorch (via tensor subclass)
- Run it faster than llama.cpp

TODO:
- [ ] Support Q8_0, Q4_0, Q4_1
- [ ] Support K-quant

## Usage

Chat demo via vLLM

```bash
python chat_vllm.py --model Llama-3.2-1B-Instruct-BF16.gguf --tokenizer meta-llama/Llama-3.2-1B-Instruct
```
