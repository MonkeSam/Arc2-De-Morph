import argparse
import os
import pandas as pd
from common.const import THRESHOLD


parser = argparse.ArgumentParser(description="Gate checker")
parser.add_argument("--csv", type=str, default="./distance_stats/", required=True, help="Csv files directory")
parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Face distance threshold")
parser.add_argument("--output", type=str, default="gate_results.log",
                    help="File where the results are written (default: gate_results.log)")
args = parser.parse_args()


def gate_passed(threshold: float, df: pd.DataFrame, out):
    print(f"Gate passed (lower is better)\tthreshold: {threshold}\n", file=out)
    test = (
        "M_vs_C_distance",
        "M_vs_A_distance",
        "DC_vs_A_distance",
        "DA_vs_C_distance"
    )
    for key in test:
        print(key, file=out)
        passed_files = df[df[key] <= threshold][["morph", key]]
        passed = passed_files['morph'].count()
        percentile = (100 * passed) / len(df.index)
        print(f"{passed} are verified \nSUCCEEDED {percentile:.2f}%\n", file=out)


def get_efficency(threshold: float, df: pd.DataFrame, out):
    print("Checking extracted faces...", file=out)
    test = (
        "DA_vs_A_distance",
        "DC_vs_C_distance"
    )
    for key in test:
        verified = df[df[key] <= threshold]
        percentile = (100 * verified[key].count()) / len(df.index)
        label = " ".join(key.split('_')[:3])
        print(f"{label} efficency = {percentile:.2f}%", file=out)

    verified = df.where(df[test[0]] < threshold).where(df[test[1]] < threshold)
    percentile = (100 * verified["morph"].count()) / len(df.index)
    print(f"\nGlobal efficency is {percentile:.2f}%\n", file=out)


out_dir = os.path.dirname(os.path.abspath(args.output))
os.makedirs(out_dir, exist_ok=True)

with open(args.output, "w", encoding="utf-8") as out:
    for name in sorted(os.listdir(args.csv)):
        if not name.lower().endswith(".csv"):
            continue

        print(f"---------------Analyzing {name}-------------------", file=out)
        stats = pd.read_csv(os.path.join(args.csv, name))

        gate_passed(args.threshold, stats, out)
        get_efficency(args.threshold, stats, out)

print(f"Results written to {args.output}")
