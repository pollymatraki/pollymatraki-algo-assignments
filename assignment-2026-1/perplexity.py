import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Perplexity Calculator")

    parser.add_argument("input_file", help="Input text file")
    parser.add_argument("output_file", help="Output file")

    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--n_ctx", type=int, default=2048)
    parser.add_argument("--begin_context_tokens", type=int, default=512)

    return parser.parse_args()


def main():
    args = parse_args()

    print("Input:", args.input_file)
    print("Output:", args.output_file)
    print("Stride:", args.stride)
    print("Context size:", args.n_ctx)
    print("Begin tokens:", args.begin_context_tokens)


if __name__ == "__main__":
    main()