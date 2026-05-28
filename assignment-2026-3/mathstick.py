import argparse
import json
import re


SEGMENTS = {
    0: {1, 2, 3, 4, 5, 6},
    1: {2, 3},
    2: {0, 1, 2, 4, 5},
    3: {0, 1, 2, 3, 4},
    4: {0, 2, 3, 6},
    5: {0, 1, 3, 4, 6},
    6: {0, 1, 3, 4, 5, 6},
    7: {1, 2, 3, 6},
    8: {0, 1, 2, 3, 4, 5, 6},
    9: {0, 1, 2, 3, 4, 6},
}


def parse_problem(problem):
    pattern = r'^\s*(\d+)\s*([+\-])\s*(\d+)\s*=\s*(\d+)\s*$'

    match = re.match(pattern, problem)

    if not match:
        raise ValueError("Invalid problem format")

    left = match.group(1)
    operator = match.group(2)
    right = match.group(3)
    result = match.group(4)

    return left, operator, right, result


def create_slots(left, right, result):
    digits = []

    for digit in left:
        digits.append(int(digit))

    for digit in right:
        digits.append(int(digit))

    for digit in result:
        digits.append(int(digit))

    slots = []

    for i in range(len(digits)):
        label = chr(ord("A") + i)
        slots.append({
            "label": label,
            "digit": digits[i]
        })

    return slots

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--problem", required=True)
    parser.add_argument("--max-k", type=int, default=2)

    args = parser.parse_args()

    left, operator, right, result = parse_problem(args.problem)

    slots = create_slots(left, right, result)

    print("SLOTS:")

    for slot in slots:
        print(slot["label"], "=", slot["digit"])


if __name__ == "__main__":
    main()