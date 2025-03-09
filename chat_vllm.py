import argparse
import asyncio
import os
import tempfile
import time

# silence vLLM outputs in stdout/stderr
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from gguf_pytorch import load_gguf, load_state_dict


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    args = parser.parse_args()

    if args.tokenizer is None and not args.model.endswith(".gguf"):
        args.tokenizer = args.model
    assert args.tokenizer is not None

    # TODO: create empty model from gguf metadata
    # we must use async API to stream outputs
    if args.model.endswith(".gguf"):
        assert args.tokenizer is not None
        config, state_dict = load_gguf(args.model, format="vllm")

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            engine_args = AsyncEngineArgs(
                model=tmpdir,
                tokenizer=args.tokenizer,
                enable_prefix_caching=True,
                load_format="dummy",
                enforce_eager=True,
            )
            llm = AsyncLLMEngine.from_engine_args(engine_args)

        # TODO: enable torch.compile and figure out how to re-enable CUDA graph
        device = llm.engine.model_executor.driver_worker.device
        model = llm.engine.model_executor.driver_worker.worker.model_runner.model
        load_state_dict(model, state_dict, assign=True)
        model.to(device)

    else:
        args.tokenizer = args.tokenizer or args.model
        engine_args = AsyncEngineArgs(
            model=args.model,
            tokenizer=args.tokenizer,
            enable_prefix_caching=True,
        )
        llm = AsyncLLMEngine.from_engine_args(engine_args)

    tokenizer = await llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=128_000)
    print("Ready for input")

    conversation = []

    while True:
        new_prompt = input("User input (Ctrl+C to exit): ")
        conversation.append(dict(role="user", content=new_prompt))
        model_inputs = tokenizer.apply_chat_template(conversation, tokenize=False)
        length = 0
        async for output in llm.generate(model_inputs, sampling_params, request_id=time.monotonic()):
            text = output.outputs[0].text
            print(text[length:], end="", flush=True)
            length = len(text)
        print("\n")
        conversation.append(dict(role="assistant", content=text))


if __name__ == "__main__":
    asyncio.run(main())
