# Arc2(De)Morph

This is a [MI@Biolab](https://miatbiolab.csr.unibo.it/) research project which has the goal to extract faces and analyzing results.
Faces are extracted from images generated using [Arc2Morph](https://github.com/MI-BioLab/arc2morph.git) project.

## Install

Demorphing and analysis work on different enviroments.

### Demorphing

You must clone [Arc2Morph](https://github.com/MI-BioLab/arc2morph.git) repo with the command

```
git clone https://github.com/MI-BioLab/arc2morph.git
```

and follow the given instruction in the README file to setup [Arc2Face](https://github.com/foivospar/Arc2Face.git).

### Analysis

For face analysis it's needed [DeepFace](https://github.com/serengil/deepface), and a new enviroment to avoid dependencies conflicts with ArcFace.

```
conda create -n deepface
conda activate deepface
git clone https://github.com/serengil/deepface.git
```

## Usage

Project is made by 4 scripts

- `demorph.py` extracts faces from a given set of image files
- `distance.py` get stats from `demorph.py` output
- `gate_check.py` summarizes stats obtained with `distance.py`

> To generate morphed images, the usage of `morphed_multiple.py` is specified in [Arc2Morph](https://github.com/MI-BioLab/arc2morph.git) repo.

### Demorph

> Execution must be in Arc2Face enviroment

Run it in the Arc2Face environment as:

```
python demorph.py --morphs <morphed_imgs_path> --original <original_imgs_path> --output <output_dir_path> --pose <face_pose_used_to_extract>
```

> Are provided same flags as Arc2Morph project

To demorph is possible to specify the face pose to extract, by default is set to `1`.\

> Interpolation is by default is set to `encoded-slerp`.

### Distance

> Execution must be in DeepFace enviroment
> Run analysis and save stats on `.csv` files in the given output path.

```
python distance.py --morphs <morphed_imgs_path> --original <original_imgs_path> --extracted <extracted_imgs_path> --output <output_path>
```

### Gate check

> Execution must be in Arc2Face enviroment

Run summary from stats saved on `.csv`.

```
python gate_check.py --csv <distance_stats_path>
```

It is possible to specify the passed gate threshold using `--threshold`, by default is `0.5143`.
