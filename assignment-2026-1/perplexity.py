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

    stride = args.stride
    n_ctx = args.n_ctx

    total_nll = 0.0
    total_tokens = 0
    bos = tokenizer.bos_token_id

    for i in range(0, len(tokens), stride):
      if i == 0:
          begin = 0
          end = min(n_ctx, len(tokens))
          target_start = args.begin_context_tokens
      else:
          begin = max(i + stride - n_ctx, 0)
          end = min(i + stride, len(tokens))
          target_start = len(tokens[begin:i])

    window = tokens[begin:end]

    window = [bos] + window

    window_tensor = torch.tensor([window])

    with torch.no_grad():
        logits = model(window_tensor).logits

    logits = logits[0]

    for j in range(target_start, len(window) - 1):
        row = logits[j].tolist()
        target_token = window[j + 1]

        max_val = max(row)
        shifted = [x - max_val for x in row]

        log_sum_exp = math.log(sum(math.exp(x) for x in shifted))

        log_probs = [x - log_sum_exp for x in shifted]

        log_prob = log_probs[target_token]

        total_nll += -log_prob
        total_tokens += 1



    nll = total_nll / total_tokens
    perplexity = math.exp(nll)

    print(perplexity)

    with open(args.output_file, "w") as f:
        f.write(str(perplexity))

if __name__ == "__main__":
    main()