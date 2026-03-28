def main():
    import math

    args = parse_args()

    # 1. Read file
    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()

    print("Loaded text length:", len(text))

    # 2. Load model + tokenizer
    model_name = "facebook/opt-125m"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        tie_word_embeddings=False
    )
    model.eval()

    # 3. Tokenization
    tokens = tokenizer(text).input_ids

    print("Total tokens:", len(tokens))
    print("First 10 tokens:", tokens[:10])

    # 4. Πάρε ένα μικρό window (για δοκιμή)
    window = tokens[:min(50, len(tokens))]

    window_tensor = torch.tensor([window])

    with torch.no_grad():
        logits = model(window_tensor).logits

    logits = logits[0]

    # 5. Υπολογισμός log probabilities
    total_log_prob = 0.0
    count = 0

    for i in range(len(window) - 1):
        row = logits[i].tolist()
        target_token = window[i + 1]

        # numerical stability
        max_val = max(row)
        shifted = [x - max_val for x in row]

        log_sum_exp = math.log(sum(math.exp(x) for x in shifted))

        log_probs = [x - log_sum_exp for x in shifted]

        log_prob = log_probs[target_token]

        total_log_prob += log_prob
        count += 1

    print("Total log prob:", total_log_prob)
    print("Count:", count)