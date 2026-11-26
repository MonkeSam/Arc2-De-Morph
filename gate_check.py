import argparse
import os
import pandas as pd
from common.const import THRESHOLD


parser = argparse.ArgumentParser(description="Gate checker")
parser.add_argument("--csv", type=str, default="./distance_stats/", required=True, help="Csv files directory")
parser.add_argument("--output", type=str, default="gate_results.log",
                    help="File where the results are written (default: gate_results.log)")
args = parser.parse_args()


def gate_passed(df: pd.DataFrame, out):
    print(f"Gate passed (lower is better)\tthreshold set during distance evaluation\n", file=out)
    test = (
        "M_vs_C_verified",
        "M_vs_A_verified",
        "DC_vs_A_verified",
        "DA_vs_C_verified"
    )
    for key in test:
        print(key, file=out)
        passed_files = df[df[key] == True][["morph", key]]
        passed = passed_files['morph'].count()
        percentile = (100 * passed) / len(df.index)
        print(f"{passed} are verified \nSUCCEEDED {percentile:.2f}%\n", file=out)


def get_efficency(df: pd.DataFrame, out):
    print("Checking extracted faces...", file=out)
    test = (
        "DA_vs_A_verified",
        "DC_vs_C_verified"
    )
    for key in test:
        verified = df[df[key] == True]
        percentile = (100 * verified[key].count()) / len(df.index)
        label = " ".join(key.split('_')[:3])
        print(f"{label} efficency = {percentile:.2f}%", file=out)

    verified = df[df[[test[0], test[1]]].all(axis=1)] #Couples of images where both demorphed subjects match their originals
    percentile = (100 * verified["morph"].count()) / len(df.index)
    print(f"\nGlobal efficency is {percentile:.2f}%\t#Couples of images where both demorphed subjects match their originals\n", file=out)



fout = os.path.join(args.output,"gate_result.log")
with open(fout, "w", encoding="utf-8") as out:
    for name in sorted(os.listdir(args.csv)):
        if not name.lower().endswith(".csv"):
            continue

        print(f"---------------Analyzing {name}-------------------", file=out)
        stats = pd.read_csv(os.path.join(args.csv, name))

        gate_passed(stats, out)
        get_efficency(stats, out)
