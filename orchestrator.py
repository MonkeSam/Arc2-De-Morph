import subprocess, sys, argparse
from common.utlis import add_common_args, algorithm_code
from common.const import DEFAULT_DEMORPH_PATH
from pathlib import Path
import shutil, os

HERE = Path(__file__).resolve().parent
CONDA = os.environ.get("CONDA_EXE") or shutil.which("conda") or sys.exit("conda not found")

#TODO:
# 1. Extracted directory must be binded with demorphing output directory
# 2. Add algorithm parameter to select morphed images generated with that type of algorithm
# 3. Double-check for missing arguments
parser = argparse.ArgumentParser(description="Demorphing pipeline with analisys")
parser = add_common_args(parser, output_dir=DEFAULT_DEMORPH_PATH,output_help="Directory of the extracted images")
parser.add_argument("--num_steps", type=int, default=25, help="Number of diffusion steps")
parser.add_argument("--guidance_scale", type=float, default=3.0, help="Guidance scale for the diffusion model")
parser.add_argument("--seed", type=int, default=23, help="Random seed for the diffusion model")
parser.add_argument("--use_lcm", action='store_true', help="Use LCM for latent consistency")
parser.add_argument("--randomize_seed", action='store_true', help="Randomize the seed for each image")
parser.add_argument("--interp_mode", choices=['arcface-slerp', 'encoded-slerp'], default='encoded-slerp', help="Interpolation mode for the morphing")
parser.add_argument("--pose", type=int,  choices=[-1,1,2,3], default=-1, help="Pose Id\n-1 => all poses \n1 => -11\n2 =>-12\n3=>-13")
parser.add_argument("--stats_path", type=str,default="./distance_stats" ,help="Distance statistics directory")

args = parser.parse_args()
ALL_POSES = (1, 2, 3)
poses = list(ALL_POSES) if args.pose == -1 else [args.pose]
morph_dir = args.morphs
original_dir = args.original
output_demorph = args.output
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



STEPS = [("arc2face", demorph_cmd(pose)) for pose in poses] + [("deepface", ["python", "distance.py",
                  "--morphs", morph_dir,
                  "--original", original_dir,
                  "--extracted", str(Path(output_demorph) / algorithm_code(algorithm) / args.interp_mode),
                  "--algorithm", algorithm,
                  "--output", args.stats_path
                  ]),

    ("deepface", ["python", "gate_check.py", "--csv", args.stats_path]),
]

def run_step(env, cmd):
    cmd = [str(c) for c in cmd]
    command = [CONDA, "run", "-n", env, "--no-capture-output", *cmd]
    try:
        subprocess.run(command, check=True, cwd=HERE)
    except subprocess.CalledProcessError as e:
        print(f"{cmd[1]} failed (exit code {e.returncode}): {' '.join(cmd)}", file=sys.stderr)


if __name__ == "__main__":

    for env, cmd in STEPS:
        print(f"Running: {' '.join(cmd)}")
        run_step(env, cmd)
    print("Pipline closed")
