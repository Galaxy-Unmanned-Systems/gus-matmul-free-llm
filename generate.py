import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import time

import mmfreelm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "ridger/MMfreeLM-370M"
MAX_NEW_TOKENS = 100
NUM_RUNS = 3
INPUT_PROMPT = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, "


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()
    inputs = tokenizer(INPUT_PROMPT, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    runs: list[dict[str, float | int | str]] = []
    with torch.inference_mode():
        for run_idx in range(NUM_RUNS):
            sync_device(device)
            start = time.perf_counter()
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            sync_device(device)
            elapsed = time.perf_counter() - start
            generated_tokens = int(outputs.shape[-1] - input_ids.shape[-1])
            tok_per_sec = generated_tokens / elapsed if elapsed > 0 else float("inf")
            runs.append(
                {
                    "run": run_idx + 1,
                    "elapsed_sec": elapsed,
                    "generated_tokens": generated_tokens,
                    "tok_per_sec": tok_per_sec,
                    "text": tokenizer.batch_decode(outputs, skip_special_tokens=True)[0],
                }
            )

    mean_tok_per_sec = sum(float(run["tok_per_sec"]) for run in runs) / len(runs)
    payload = {
        "model_name": MODEL_NAME,
        "device": str(device),
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_runs": NUM_RUNS,
        "mean_tok_per_sec": mean_tok_per_sec,
        "runs": runs,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
