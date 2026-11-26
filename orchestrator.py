import subprocess, sys, argparse
from common.utlis import add_common_args, algorithm_code
from common.const import DEFAULT_OUTPUT_ROOT, DEMORPH_SUBDIR, STATS_SUBDIR, GATE_LOG_NAME
from pathlib import Path
import shutil, os

HERE = Path(__file__).resolve().parent
CONDA = os.environ.get("CONDA_EXE") or shutil.which("conda") or sys.exit("conda not found")

parser = argparse.ArgumentParser(description="Demorphing and analisys pipeline")
parser = add_common_args(parser, output_dir=DEFAULT_OUTPUT_ROOT, output_required=False,
                         output_help="Root directory for all outputs (demorphed images and distance statistics)")
parser.add_argument("--num_steps", type=int, default=25, help="Number of diffusion steps")
parser.add_argument("--guidance_scale", type=float, default=3.0, help="Guidance scale for the diffusion model")
parser.add_argument("--seed", type=int, default=23, help="Random seed for the diffusion model")
parser.add_argument("--use_lcm", action='store_true', help="Use LCM for latent consistency")
parser.add_argument("--randomize_seed", action='store_true', help="Randomize the seed for each image")
parser.add_argument("--interp_mode", choices=['arcface-slerp', 'encoded-slerp'], default='encoded-slerp', help="Interpolation mode for the morphing")
parser.add_argument("--pose", type=int,  choices=[-1,1,2,3], default=-1, help="Pose Id\n-1 => all poses \n1 => -11\n2 =>-12\n3=>-13")

args = parser.parse_args()
ALL_POSES = (1, 2, 3)
poses = list(ALL_POSES) if args.pose == -1 else [args.pose]
morph_dir = args.morphs
original_dir = args.original
# Relative paths are resolved from this script's folder, like the child scripts (cwd=HERE) do
output_root = HERE / args.output
output_demorph = str(output_root / DEMORPH_SUBDIR)
stats_path = str(output_root / STATS_SUBDIR)
for d in (output_demorph, stats_path):
    Path(d).mkdir(parents=True, exist_ok=True)
algorithm = args.algorithm

def demorph_cmd(pose):
    return ["python", "demorph.py",
            "--algorithm", algorithm,
            "--morphs", morph_dir,
            "--original", original_dir,
            "--output", output_demorph,
            "--num_steps", str(args.num_steps),
            "--guidance_scale", str(args.guidance_scale),
            "--seed", str(args.seed),
            "--interp_mode", args.interp_mode,
            "--pose", str(pose),
            *(["--use_lcm"] if args.use_lcm else []),
            *(["--randomize_seed"] if args.randomize_seed else [])
            ]



STEPS = [("arc2face", demorph_cmd(pose), None) for pose in poses] + [("deepface", ["python", "distance.py",
                  "--morphs", morph_dir,
                  "--original", original_dir,
                  "--extracted", str(Path(output_demorph) / algorithm_code(algorithm) / args.interp_mode),
                  "--algorithm", algorithm,
                  "--output", stats_path
                  ], None),

    ("deepface", ["python", "gate_check.py", "--csv", stats_path, "--output", str(output_root)], None),
]

def run_step(env, cmd, log_path=None):
    cmd = [str(c) for c in cmd]
    command = [CONDA, "run", "-n", env, "--no-capture-output", *cmd]
    try:
        if log_path is None:
            subprocess.run(command, check=True, cwd=HERE)
        else:
            # Show the output and also save it to the log file
            with open(log_path, "w", encoding="utf-8") as log, \
                 subprocess.Popen(command, cwd=HERE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) as proc:
                for line in proc.stdout:
                    print(line, end="")
                    log.write(line)
            if proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode, command)
    except subprocess.CalledProcessError as e:
        print(f"{cmd[1]} failed (exit code {e.returncode}): {' '.join(cmd)}", file=sys.stderr)


if __name__ == "__main__":

    for env, cmd, log_path in STEPS:
        print(f"Running: {' '.join(cmd)}")
        run_step(env, cmd, log_path)
    print("Pipline closed")
