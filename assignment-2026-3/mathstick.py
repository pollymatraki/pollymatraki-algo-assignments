import argparse
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
    pattern = r"^\s*(\d+)\s*([+\-])\s*(\d+)\s*=\s*(\d+)\s*$"
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


def get_stick_labels(slot_label, digit):
    labels = []

    for segment in sorted(SEGMENTS[digit]):
        labels.append(slot_label + str(segment))

    return labels


def get_digit_change(source_digit, target_digit):
    source_segments = SEGMENTS[source_digit]
    target_segments = SEGMENTS[target_digit]

    added_segments = target_segments - source_segments
    removed_segments = source_segments - target_segments

    return added_segments, removed_segments


def get_candidates_for_slot(slot, max_k):
    source_digit = slot["digit"]
    slot_label = slot["label"]

    candidates = []

    for target_digit in range(10):
        added_segments, removed_segments = get_digit_change(source_digit, target_digit)

        if len(added_segments) <= max_k and len(removed_segments) <= max_k:
            added_labels = []
            removed_labels = []

            for segment in sorted(added_segments):
                added_labels.append(slot_label + str(segment))

            for segment in sorted(removed_segments):
                removed_labels.append(slot_label + str(segment))

            candidates.append({
                "target_digit": target_digit,
                "added": added_labels,
                "removed": removed_labels,
                "delta": len(added_labels) - len(removed_labels)
            })

    return candidates


def print_slots(slots):
    print("SLOTS:")

    for slot in slots:
        print(slot["label"], "=", slot["digit"])


def print_candidates(slots, max_k):
    print()
    print("CANDIDATES:")

    for slot in slots:
        print("Slot", slot["label"], "digit", slot["digit"])

        candidates = get_candidates_for_slot(slot, max_k)

        for candidate in candidates:
            print(
                "  ->",
                candidate["target_digit"],
                "added:",
                candidate["added"],
                "removed:",
                candidate["removed"],
                "delta:",
                candidate["delta"]
            )

def build_digit_transitions(max_k):
    transitions = {}

    for source_digit in range(10):
        transitions[source_digit] = []

        for target_digit in range(10):
            source_segments = SEGMENTS[source_digit]
            target_segments = SEGMENTS[target_digit]

            added_segments = target_segments - source_segments
            removed_segments = source_segments - target_segments

            a = len(added_segments)
            r = len(removed_segments)

            if a <= max_k and r <= max_k:
                transitions[source_digit].append({
                    "target_digit": target_digit,
                    "added_segments": sorted(added_segments),
                    "removed_segments": sorted(removed_segments),
                    "a": a,
                    "r": r,
                    "delta": a - r
                })

    return transitions


def print_digit_transitions(transitions, digit):
    print("TRANSITIONS FOR DIGIT", digit)

    for transition in transitions[digit]:
        print(
            digit,
            "->",
            transition["target_digit"],
            "added:",
            transition["added_segments"],
            "removed:",
            transition["removed_segments"],
            "a:",
            transition["a"],
            "r:",
            transition["r"],
            "delta:",
            transition["delta"]
        )

def compute_slot_delta_intervals(slots, transitions):
    intervals = []

    for slot in slots:
        digit = slot["digit"]
        digit_transitions = transitions[digit]

        min_delta = digit_transitions[0]["delta"]
        max_delta = digit_transitions[0]["delta"]

        for transition in digit_transitions:
            if transition["delta"] < min_delta:
                min_delta = transition["delta"]

            if transition["delta"] > max_delta:
                max_delta = transition["delta"]

        intervals.append({
            "slot": slot["label"],
            "digit": digit,
            "min_delta": min_delta,
            "max_delta": max_delta
        })

    return intervals


def print_slot_delta_intervals(intervals):
    print()
    print("SLOT DELTA INTERVALS:")

    for item in intervals:
        print(
            item["slot"],
            "digit",
            item["digit"],
            "[",
            item["min_delta"],
            ",",
            item["max_delta"],
            "]"
        )
def compute_suffix_intervals(intervals):
    suffixes = []

    suffix_min = 0
    suffix_max = 0

    for i in range(len(intervals) - 1, -1, -1):
        suffix_min += intervals[i]["min_delta"]
        suffix_max += intervals[i]["max_delta"]

        suffixes.insert(0, {
            "index": i,
            "slot": intervals[i]["slot"],
            "suffix_min": suffix_min,
            "suffix_max": suffix_max
        })

    return suffixes


def print_suffix_intervals(suffixes):
    print()
    print("SUFFIX INTERVALS:")

    for item in suffixes:
        print(
            item["slot"],
            "[",
            item["suffix_min"],
            ",",
            item["suffix_max"],
            "]"
        )

def get_operator_transition(source_operator, target_operator):
    if source_operator == target_operator:
        return {
            "target_operator": target_operator,
            "added": [],
            "removed": [],
            "oa": 0,
            "or": 0,
            "od": 0
        }

    if source_operator == "-" and target_operator == "+":
        return {
            "target_operator": target_operator,
            "added": ["G0"],
            "removed": [],
            "oa": 1,
            "or": 0,
            "od": -1
        }

    return {
        "target_operator": target_operator,
        "added": [],
        "removed": ["G0"],
        "oa": 0,
        "or": 1,
        "od": 1
    }


def print_operator_transitions(operator):
    print()
    print("OPERATOR TRANSITIONS:")

    for target_operator in ["+", "-"]:
        transition = get_operator_transition(operator, target_operator)

        print(
            operator,
            "->",
            target_operator,
            "added:",
            transition["added"],
            "removed:",
            transition["removed"],
            "oa:",
            transition["oa"],
            "or:",
            transition["or"],
            "od:",
            transition["od"]
        )

def create_empty_output(problem, max_k):
    output = {
        "problem": problem,
        "max_k": max_k,
        "counts": {},
        "nodes_visited": 0,
        "nodes_pruned": 0,
        "solutions": {}
    }

    for k in range(1, max_k + 1):
        output["counts"][str(k)] = 0
        output["solutions"][str(k)] = []

    return output

def create_search_state():
    return {
        "nodes_visited": 0,
        "nodes_pruned": 0,
        "solutions": []
    }
def do_slot(
    index,
    slots,
    transitions,
    suffixes,
    operator_transition,
    max_k,
    current_solution,
    total_added,
    total_removed,
    state,
    left,
    right,
    result
):
    state["nodes_visited"] += 1

    if index == len(slots):
        final_added = total_added + operator_transition["oa"]
        final_removed = total_removed + operator_transition["or"]

        if final_added == final_removed:
            if is_valid_equation(
                current_solution,
                left,
                right,
                result,
                operator_transition["target_operator"]
            ):
                state["solutions"].append(current_solution.copy())

            return

    current_digit = slots[index]["digit"]

    for transition in transitions[current_digit]:
        new_total_added = total_added + transition["a"]
        new_total_removed = total_removed + transition["r"]

        if new_total_added + operator_transition["oa"] > max_k:
            state["nodes_pruned"] += 1
            continue

        if new_total_removed + operator_transition["or"] > max_k:
            state["nodes_pruned"] += 1
            continue

        n = operator_transition["od"] - (new_total_added - new_total_removed)

        if index + 1 < len(slots):
            suffix_min = suffixes[index + 1]["suffix_min"]
            suffix_max = suffixes[index + 1]["suffix_max"]
        else:
            suffix_min = 0
            suffix_max = 0

        if n < suffix_min or n > suffix_max:
            state["nodes_pruned"] += 1
            continue

        current_solution.append(transition["target_digit"])

        do_slot(
            index + 1,
            slots,
            transitions,
            suffixes,
            operator_transition,
            max_k,
            current_solution,
            new_total_added,
            new_total_removed,
            state,
            left,
            right,
            result
        )

        current_solution.pop()

def digits_to_number(digits):
    value = 0

    for digit in digits:
        value = value * 10 + digit

    return value


def split_solution_digits(solution_digits, left, right, result):
    left_len = len(left)
    right_len = len(right)
    result_len = len(result)

    left_digits = solution_digits[0:left_len]
    right_digits = solution_digits[left_len:left_len + right_len]
    result_digits = solution_digits[left_len + right_len:left_len + right_len + result_len]

    return left_digits, right_digits, result_digits


def is_valid_equation(solution_digits, left, right, result, operator):
    left_digits, right_digits, result_digits = split_solution_digits(
        solution_digits,
        left,
        right,
        result
    )

    left_value = digits_to_number(left_digits)
    right_value = digits_to_number(right_digits)
    result_value = digits_to_number(result_digits)

    if operator == "+":
        return left_value + right_value == result_value

    return left_value - right_value == result_value

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--problem", required=True)
    parser.add_argument("--max-k", type=int, default=2)

    args = parser.parse_args()

    left, operator, right, result = parse_problem(args.problem)

    slots = create_slots(left, right, result)

    print_slots(slots)
    print_candidates(slots, args.max_k)
    transitions = build_digit_transitions(args.max_k)

    intervals = compute_slot_delta_intervals(slots, transitions)
    print_slot_delta_intervals(intervals)
    

    suffixes = compute_suffix_intervals(intervals)
    print_suffix_intervals(suffixes)

    print_operator_transitions(operator)
    
    print()
    print_digit_transitions(transitions, 0)

    print()
    print_digit_transitions(transitions, 1)
    output = create_empty_output(args.problem, args.max_k)

    print()
    print("JSON OUTPUT STRUCTURE")
    print(output)

    
    state = create_search_state()

    operator_transition = get_operator_transition(operator, operator)

    do_slot(
        0,
        slots,
        transitions,
        suffixes,
        operator_transition,
        args.max_k,
        [],
        0,
        0,
        state,
        left,
        right,
        result
    )

    print()
    print("DFS TEST")
    print("Visited:", state["nodes_visited"])
    print("Solutions:", len(state["solutions"]))

    print("Pruned:", state["nodes_pruned"])
    for solution in state["solutions"]:
        print(solution)
    print()

    
if __name__ == "__main__":
    main()
    