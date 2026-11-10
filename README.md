# Arc2(De)Morph

This is an [MI@Biolab](https://miatbiolab.csr.unibo.it/) research project aimed at extracting faces and analyzing the results. Faces are extracted from images of the [FEI Morph Dataset](https://miatbiolab.csr.unibo.it/fei-morph-dataset/), generated using the [Arc2Morph](https://github.com/MI-BioLab/arc2morph.git) project.

## Installation

Demorphing and analysis require different environments.

### Demorphing

You must clone the [Arc2Morph](https://github.com/MI-BioLab/arc2morph.git) repository with the following command:

```bash
git clone https://github.com/MI-BioLab/arc2morph.git
```

and follow the instructions provided in the README.md file to set up [Arc2Face](https://github.com/foivospar/Arc2Face.git).

### Analysis

Face analysis requires [DeepFace](https://github.com/serengil/deepface) and a new environment to avoid dependency conflicts with ArcFace.

```bash
conda create -n deepface
conda activate deepface
pip install deepface
pip install pandas # we use pandas to analyze DeepFace results
```

## Usage

The project consists of 3 scripts:

- `demorph.py` extracts faces from a given set of image files.
- `distance.py` gets statistics from the `demorph.py` output.
- `gate_check.py` summarizes the statistics obtained with `distance.py`.

> To generate morphed images, instructions for using `morphed_multiple.py` are specified in the [Arc2Morph](https://github.com/MI-BioLab/arc2morph.git) repository.

### Orchestrator

The demorphing and analysis pipeline is executed by `orchestrator.py`, which runs every stage of the project.

```bash
python orchestrator.py --morphs <morphed_imgs_dir> --original <original_imgs_dir> --algorithm <algorithm_code> --stats_path <save_stats_dir> --output <demorphed_imgs_output_dir>
```

Note:
- The `--algorithm` flag requires the code of the algorithm used to morph faces, e.g., *C05, C24, ...* ([Arc2Morph](https://github.com/MI-BioLab/arc2morph.git)).
- By default, the `--pose` flag is set to `-1`, so each face is extracted using every available pose (11, 12, 13).
- The orchestrator runs each script in its own environment.

### Run the scripts individually

Scripts can also be executed individually instead of using `orchestrator.py`.

#### Demorph

> Execution must take place in the Arc2Face environment.

Run it in the Arc2Face environment as follows:

```bash
python demorph.py --morphs <morphed_imgs_path> --original <original_imgs_path> --algorithm <algorithm_code> --output <output_dir_path> --pose <face_pose_used_to_extract>
```

> The same flags as in the Arc2Morph project are provided.

For demorphing, it is possible to specify the face pose to extract; by default, this is set to `1`.

> Interpolation is set to `encoded-slerp` by default.

#### Distance

> Execution must take place in the DeepFace environment.
> Run the analysis and save the statistics to `.csv` files in the specified output path.

```bash
python distance.py --morphs <morphed_imgs_path> --original <original_imgs_path> --algorithm <algorithm_code> --extracted <extracted_imgs_path> --output <output_path>
```

#### Gate check

> Execution must take place in the DeepFace environment.

Run a summary using the statistics saved in the `.csv` files.

```bash
python gate_check.py --csv <distance_stats_path>
```

It is possible to specify the gate pass threshold using `--threshold`; the default value is `0.5143`.
