import argparse
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Perplexity Calculator")

    parser.add_argument("input_file")
    parser.add_argument("output_file")

    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--n_ctx", type=int, default=2048)
    parser.add_argument("--begin_context_tokens", type=int, default=512)

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Read file
    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()

    print("Loaded text length:", len(text))

    # 2. Tokenizer
    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokens = tokenizer(text).input_ids

    print("Total tokens:", len(tokens))
    print("First 10 tokens:", tokens[:10])


if __name__ == "__main__":
    main()