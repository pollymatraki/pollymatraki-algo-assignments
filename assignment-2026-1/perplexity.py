import argparse
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Perplexity Calculator")
    parser.add_argument("input_file")
    parser.add_argument("out_file")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--n-ctx", dest="n_ctx", type=int, default=2048)
    parser.add_argument(
        "--begin-context-tokens",
        dest="begin_context_tokens",
        type=int,
        default=512,
    )
    return parser.parse_args()


def log_probs_from_row(row):
    max_val = max(row)
    shifted = [x - max_val for x in row]
    log_sum_exp = math.log(sum(math.exp(x) for x in shifted))
    return [x - log_sum_exp for x in shifted]


def main():
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
    bos_token = tokenizer.bos_token_id

    stride = args.stride
    n_ctx = args.n_ctx
    begin_context_tokens = args.begin_context_tokens

    max_window_tokens = n_ctx - 1

    total_nll = 0.0
    total_predicted_tokens = 0

    for i in range(0, len(tokens), stride):
        if i == 0:
            begin = 0
            end = min(begin_context_tokens + stride, len(tokens))
            end = min(end, max_window_tokens)
            context_len = begin_context_tokens
        else:
            begin = max(i + stride - max_window_tokens, 0)
            end = min(i + stride, len(tokens))
            context_len = i - begin

        window = tokens[begin:end]
        window_with_bos = [bos_token] + window

        window_tensor = torch.tensor([window_with_bos])

        with torch.no_grad():
            logits = model(window_tensor).logits

        logits = logits[0]

        start_j = context_len
        end_j = len(window_with_bos) - 1

        for j in range(start_j, end_j):
            row = logits[j].tolist()
            target_token = window_with_bos[j + 1]

            log_probs = log_probs_from_row(row)
            log_prob = log_probs[target_token]

            total_nll += -log_prob
            total_predicted_tokens += 1

    nll = total_nll / total_predicted_tokens
    perplexity = math.exp(nll)

    with open(args.out_file, "w", encoding="utf-8") as f:
        f.write(f"{perplexity}\n")


if __name__ == "__main__":
    main()