import subprocess, sys, argparse

#TODO:
# 1. Extracted directory must be binded with demorphing output directory
# 2. Add algorithm parameter to select morphed images generated with that type of algorithm
# 3. Double-check for missing arguments
parser = argparse.ArgumentParser(description="Demorphing pipeline with analisys")
parser.add_argument("--morphs", type=str, required=True, help="Directory of the morphed images")
parser.add_argument("--original", type=str, required=True, help="Directory of the original images")
parser.add_argument("--num_steps", type=int, default=25, help="Number of diffusion steps")
parser.add_argument("--guidance_scale", type=float, default=3.0, help="Guidance scale for the diffusion model")
parser.add_argument("--seed", type=int, default=23, help="Random seed for the diffusion model")
parser.add_argument("--use_lcm", action='store_true', help="Use LCM for latent consistency")
parser.add_argument("--randomize_seed", action='store_true', help="Randomize the seed for each image")
parser.add_argument("--interp_mode", choices=['arcface-slerp', 'encoded-slerp'], default='encoded-slerp', help="Interpolation mode for the morphing")
parser.add_argument("--pose", type=int,  choices=[1,2,3], default=1, help="Pose Id \n1 => -11\n2 =>-12\n3=>-13")
parser.add_argument("--extracted", type=str, required=True, help="Directory of the extracted images")
parser.add_argument("--algorithm", type=int, required=True, help="Select images generated with a specific algorithm, must be algorithm integer code for \'CXX\'")
args = parser.parse_args()


STEP = [
    ("arc2face", ["python","demorph.py"]),
    ("deepface", ["python","distance.py"]),
    ("deepface", ["python","gate_check.py"])
]

def run_step(env, cmd):
    command = ["conda", "run", "-n", env, "--no-capture-output", *cmd]
    try:
        subprocess.run(command,check=True)
    except subprocess.CalledProcessError as e:
        print(f"{cmd[-1]} failed! (exit code {e.returncode}): {' '.join(cmd)}", file=sys.stderr)
