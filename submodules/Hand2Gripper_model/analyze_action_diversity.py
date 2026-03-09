"""
Analyze the diversity of GT L/R index combinations in the dataset.

selected_gripper_blr_ids: (3,)
  - [0]: gripper type id (B)
  - [1]: L index
  - [2]: R index

Treat (L, R) as an unordered pair, e.g. (4,8) and (8,4) are counted as the same class.
Group statistics by action type (directory name) and output overall statistics.
"""

import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def analyze_action_diversity(dataset_dir):
    """
    Traverse all .npz files under dataset_dir,
    count the occurrences of (L, R) unordered pairs for each action type.
    """
    # Group by action type: action_type -> { (min, max) pair: count }
    action_pair_counts = defaultdict(lambda: defaultdict(int))
    # Overall statistics
    overall_pair_counts = defaultdict(int)
    # Total sample count
    total_samples = 0
    # Failed files
    failed_files = []

    # Collect all .npz files
    all_npz_files = []
    for dirpath, dirnames, filenames in os.walk(dataset_dir):
        for fn in filenames:
            if fn.lower().endswith('.npz'):
                all_npz_files.append(os.path.join(dirpath, fn))

    all_npz_files.sort()
    print(f"Found {len(all_npz_files)} .npz files")

    for fpath in tqdm(all_npz_files, desc="Analyzing"):
        try:
            data = np.load(fpath, allow_pickle=True)
            if "selected_gripper_blr_ids" not in data:
                continue

            blr = data["selected_gripper_blr_ids"]  # (3,)
            l_idx = int(blr[1])
            r_idx = int(blr[2])

            # Unordered pair: use frozenset or sorted tuple
            pair = tuple(sorted((l_idx, r_idx)))

            # Infer action type: extract from path
            # Path format may be: .../action_type/object_XX/left_or_right/xxx.npz
            # Or:              .../train/xxx.npz (flat)
            rel_path = os.path.relpath(fpath, dataset_dir)
            parts = rel_path.split(os.sep)

            if len(parts) >= 2:
                action_type = parts[0]  # first-level directory as action type
            else:
                action_type = "unknown"

            action_pair_counts[action_type][pair] += 1
            overall_pair_counts[pair] += 1
            total_samples += 1

        except Exception as e:
            failed_files.append((fpath, str(e)))

    # ========== Print Results ==========
    print("\n" + "=" * 70)
    print(f"Total samples: {total_samples}")
    print(f"Load failures: {len(failed_files)}")
    print("=" * 70)

    # Print per action type
    for action_type in sorted(action_pair_counts.keys()):
        pairs = action_pair_counts[action_type]
        action_total = sum(pairs.values())
        print(f"\n{'─' * 50}")
        print(f"Action type: {action_type}  (total {action_total} samples)")
        print(f"{'─' * 50}")
        print(f"  {'(L, R) unordered pair':<20} {'count':>8} {'ratio':>10}")
        for pair, count in sorted(pairs.items(), key=lambda x: -x[1]):
            pct = count / action_total * 100
            print(f"  {str(pair):<20} {count:>8} {pct:>9.1f}%")
        print(f"  Distinct pairs: {len(pairs)}")

    # Overall statistics
    print(f"\n{'=' * 70}")
    print(f"Overall (L, R) unordered pair statistics  (total {total_samples} samples)")
    print(f"{'=' * 70}")
    print(f"  {'(L, R) unordered pair':<20} {'count':>8} {'ratio':>10}")
    for pair, count in sorted(overall_pair_counts.items(), key=lambda x: -x[1]):
        pct = count / total_samples * 100
        print(f"  {str(pair):<20} {count:>8} {pct:>9.1f}%")
    print(f"\n  Total distinct (L, R) pairs: {len(overall_pair_counts)}")

    # Return result dictionary
    result = {
        "per_action": {k: dict(v) for k, v in action_pair_counts.items()},
        "overall": dict(overall_pair_counts),
        "total_samples": total_samples,
    }
    return result


if __name__ == "__main__":
    dataset_path = "/data0/Hand2GripperDatasets/Hand2Gripper_Dataset/"
    result = analyze_action_diversity(dataset_path)

    print("\n\n========== Final Result Dict ==========")
    print("per_action:")
    for action, pairs in result["per_action"].items():
        print(f"  {action}: {pairs}")
    print(f"overall: {result['overall']}")
    print(f"total_samples: {result['total_samples']}")
