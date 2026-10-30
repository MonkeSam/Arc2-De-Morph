from deepface import DeepFace
import argparse, re, os, csv,logging
from collections import defaultdict
from deepface.modules.demography import List

LABELS = ["A_vs_C", "M_vs_C", "M_vs_A", "DC_vs_A", "DC_vs_C", "DA_vs_C", "DA_vs_A"]
FIELDS = ["morph","pose", "criminal", "accomplice"] + \
         [f"{l}_{k}" for l in LABELS for k in ("distance", "verified")]
POSES = ["11","12","13"]

ALGORITM = "C05"

def get_stats(morphed_path,criminal,accomplice,demorph_criminal,demorph_accomplice,log_file):
    '''
    Analyzing  every combination distance with DeepFace.
    '''
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        encoding="utf-8",
    )
    logger = logging.getLogger(__name__)
    compare = {
        "A_vs_C": (accomplice, criminal),
        "M_vs_C": (morphed_path, criminal),
        "M_vs_A": (morphed_path, accomplice),
        "DC_vs_A": (demorph_criminal, accomplice),
        "DC_vs_C": (demorph_criminal, criminal),
        "DA_vs_C": (demorph_accomplice, criminal),
        "DA_vs_A": (demorph_accomplice, accomplice)
    }
    results = {}
    for c in compare:
        try:
            stats : dict = DeepFace.verify(img1_path = compare[c][0], img2_path = compare[c][1], model_name="ArcFace", detector_backend ='mtcnn')
            results[c] = stats
        except ValueError as e:
            message = f"{e} calculating {c} with img1_path = {compare[c][0]} and img2_path = {compare[c][1]}"
            print(message)
            logger.error(message)

    return results

def load_done(output_file):
    '''
    Returns the morph files that are already listed in the CSV file.
    '''
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return set()
    done = set()
    with open(output_file, newline="") as f:
        for row in csv.DictReader(f):

            if row.get("morph") and row.get(FIELDS[-1]) not in (None, ""):
                done.add(row["morph"])
    return done

def init_csv(output_file):
    '''
    Initializing csv file with header if absent
    '''
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        with open(output_file, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def append_row(entry, output_file):
    '''
    Append to the csv file results returned after analysis.
    '''
    row = {
        "morph": entry["morph"],
        "pose": entry["pose"],
        "criminal": entry["criminal"],
        "accomplice": entry["accomplice"]
    }
    for l in LABELS:
        r = entry["stats"].get(l)
        row[f"{l}_distance"] = r["distance"] if r else ""
        row[f"{l}_verified"] = r["verified"] if r else ""

    with open(output_file, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)


parser = argparse.ArgumentParser(description="Demorphed distance")

parser.add_argument("--morphs", type=str, required=True, help="Directory of the morphed images")
parser.add_argument("--original", type=str, required=True, help="Directory of the original images")
parser.add_argument("--extracted", type=str, required=True, help="Directory of the extracted images")
parser.add_argument("--output", type=str, required=True, help="Output directory for stats.csv")
args = parser.parse_args()

morph_pattern = re.compile(rf"^(M_([^_]+)_([^_]+)_{ALGORITM}_.+)\.png$")
inv_pattern   = re.compile(r"^INV_(M_.+)_REMOVED_([^_-]+)-([^_]+)\.png$")

inv_by_stem = defaultdict(list)
for name in sorted(os.listdir(args.extracted)):
    m = inv_pattern.match(name)
    if m:
        stem, removed_subj, pose = m.groups()
        inv_by_stem[(stem,pose)].append({"path": os.path.join(args.extracted,name),"removed":removed_subj})


paths = []
'''
For each morphed image are extracted usefull data to identify subjects,
then for each pose are stored paths and names needed for image analysis.
'''
for name in sorted(os.listdir(args.morphs)):
    m = morph_pattern.match(name)
    if not m:
        continue
    stem, crim_id, acc_id= m.groups()
    crim_id = crim_id.split("-")[0]
    acc_id = acc_id.split("-")[0]

    inv_files = {}
    for pose in POSES:
        inv_files[pose] = inv_by_stem.get((stem,pose), [])
        criminal = f"{crim_id}-{pose}.jpg"
        accomplice =  f"{acc_id}-{pose}.jpg"

        r_crim = inv_files[pose][0]["removed"]
        r_acc = inv_files[pose][1]["removed"]
        demorphed_criminal = inv_files[pose][1]["path"]
        demorphed_accomplice = inv_files[pose][0]["path"]
        if r_crim != crim_id: #asserting removed criminal is the first of the list, if is not it will be swapped
            r_acc = r_crim
            r_crim = inv_files[pose][1]["removed"]

            demorphed_criminal = inv_files[pose][0]["path"] #demorphed_crimnal is the image in which accomplice was removed
            demorphed_accomplice = inv_files[pose][1]["path"] #demorphed_accomplice same as demorphed_criminal


        paths.append({
            "morph_stem": stem,
            "pose": pose,
            "morphed": os.path.join(args.morphs, name),
            "criminal": os.path.join(args.original,criminal),
            "accomplice": os.path.join(args.original,accomplice),
            "demorphed_criminal": demorphed_criminal,
            "demorphed_accomplice": demorphed_accomplice
        })



    if not inv_files:
        continue
fname =f"stats_{os.path.basename(os.path.normpath(args.extracted))}_{ALGORITM}"
log_file = f"LOG_{fname}.log"
out_file = os.path.join(args.output, fname + ".csv")
init_csv(out_file)
done = load_done(out_file)
print(f"{len(done)} morph analyzed, {len(paths) - len(done)} not analyzed")

'''
Every stored data (paths and names) are passed to the analysis function and the results
will be stored in a .csv file appending stats returned.
'''
for p in paths:
    if p["morph_stem"] in done:
        continue
    stats = get_stats(p["morphed"] ,p["criminal"], p["accomplice"],
                        p["demorphed_criminal"], p["demorphed_accomplice"],log_file)
    append_row({
        "morph": p["morph_stem"],
        "pose": p["pose"],
        "criminal": os.path.splitext(os.path.basename(p["criminal"]))[0],
        "accomplice": os.path.splitext(os.path.basename(p["accomplice"]))[0],
        "stats": stats,
    }, out_file)
