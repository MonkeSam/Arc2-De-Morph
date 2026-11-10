import argparse,os
from pstats import Stats
import pandas as pd

THRESHOLD = 0.5143

parser = argparse.ArgumentParser(description="Gate checker")

#parser.add_argument("--output", type=str, required=True, help="Output directory for ")
parser.add_argument("--csv", type=str,required=True, default="", help ="Csv files directory")
parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Face distance threshold")
args = parser.parse_args()



def gate_passed(threshold:float,df:pd.DataFrame):
    print(f"Gate passed (lower is better)\tthreshold: {threshold}\n")
    test = (
        "M_vs_C_distance",
        "M_vs_A_distance",
        "DC_vs_A_distance",
        "DA_vs_C_distance"
    )
    for key in test:
        print(key)
        passed_files = df[df[key] <= threshold][["morph",key]]
        passed=passed_files['morph'].count()
        percentile = (100*passed)/len(df.index)
        print(f"{passed} are verified \nSUCCEEDED {percentile:.2f}%\n")

def get_efficency(threshold:float,df:pd.DataFrame):
    print("Checking extracted faces...")
    test = (
        "DA_vs_A_distance",
        "DC_vs_C_distance"
    )
    for key in test:
        verified = df[df[key] <= threshold]
        percentile = (100*verified[key].count())/len(df.index)
        label = f"{key.split('_')[0]} {key.split('_')[1]} {key.split('_')[2]}"
        print(f"{label} efficency = {percentile:.2f}%")

    verified = df.where(df[test[0]]< threshold).where(df[test[1]]<threshold)
    percentile = (100*verified["morph"].count())/len(df.index)
    print(f"\nGlobal efficency is {percentile:.2f}%")





for csv in sorted(os.listdir(args.csv)):

    print(f"---------------Analyzing {csv}-------------------")
    csv = os.path.join(args.csv,csv)
    stats = pd.read_csv(csv)

    gate_passed(args.threshold,stats)
    get_efficency(args.threshold,stats)
