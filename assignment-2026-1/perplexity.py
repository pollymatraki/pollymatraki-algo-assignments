import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Perplexity Calculator")
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--n_ctx", type=int, default=2048)
    parser.add_argument("--begin_context_tokens", type=int, default=512)
    return parser.parse_args()


def main():
    import math

    args = parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()

    model_name = "facebook/opt-125m"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        tie_word_embeddings=False
    )
    model.eval()

    tokens = tokenizer(text).input_ids

    window = tokens[:min(50, len(tokens))]

    window_tensor = torch.tensor([window])

    with torch.no_grad():
        logits = model(window_tensor).logits

    logits = logits[0]

    total_log_prob = 0.0
    count = 0

    for i in range(len(window) - 1):
        row = logits[i].tolist()
        target_token = window[i + 1]

        max_val = max(row)
        shifted = [x - max_val for x in row]

        log_sum_exp = math.log(sum(math.exp(x) for x in shifted))

        log_probs = [x - log_sum_exp for x in shifted]

        log_prob = log_probs[target_token]

        total_log_prob += log_prob
        count += 1

    print(total_log_prob)
    print(count)


if __name__ == "__main__":
    main()