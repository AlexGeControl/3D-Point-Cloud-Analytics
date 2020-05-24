# Deep Learning Point Cloud Object Detection 

KITTI 3D Object Detection evaluation, Assignment 06 of [3D Point Cloud Processing](https://www.shenlanxueyuan.com/course/204) from [深蓝学院](https://www.shenlanxueyuan.com/)

---

## Environment Setup

The solution has been tested using **xenial-gpu**. Please follow the instruction in repo root to build and run the docker instance.

The Anaconda environment is avaiable at (click to follow the link) **[/workspace/assignments/06-deep-detection/environment/deep-detection.yaml](deep-detection.yaml)**
---

## Homework Solution

---

### Setup KITTI Object Detection Evaluation Toolkit

The adapted toolkit is available at (click to follow the link) **[/workspace/assignments/06-deep-detection/kitti-eval](kitti-eval)**

Use the following commands to compile and run it:

```bash
# build:
mkdir build
cd build && make 
# run:
./kitti_eval_node /workspace/data/kitti-3d-object-detection/training/label_2/ /workspace/data/kitti-3d-object-detection/training/pred_2/
```

---

### Generate Object Detection Results Using Ground Truth

In order to generate object detection results from ground truth labels, use the following script inside (click to follow the link) **[/workspace/assignments/06-deep-detection/kitti-eval/create_pred_from_ground_truth.py](kitti-eval)**

```bash
# create object detection results from ground truth labels:
./create_pred_from_ground_truth.py -i /workspace/data/kitti-3d-object-detection/training/label_2/ -o /workspace/data/kitti-3d-object-detection/training/pred_2/
```

The Python script uses Pandas to format ground truth labels as object detection results required by KITTI evaluation toolkit:

```python
#!/opt/conda/envs/deep-detection/bin/python

import argparse

import os
import glob

import pandas as pd

import progressbar


def generate_detection_results(input_dir, output_dir):
    """ 
    Create KITTI 3D object detection results from labels

    """
    # create output dir:
    os.mkdir(
        os.path.join(output_dir, 'data')
    )

    # get input point cloud filename:
    for input_filename in progressbar.progressbar(
        glob.glob(
            os.path.join(input_dir, '*.txt')
        )
    ):
        # read data:
        label = pd.read_csv(input_filename, sep=' ', header=None)
        label.columns = [
            'category',
            'truncation', 'occlusion', 
            'alpha',
            '2d_bbox_left', '2d_bbox_top', '2d_bbox_right', '2d_bbox_bottom', 
            'height', 'width', 'length', 
            'location_x', 'location_y', 'location_z',
            'rotation'
        ]
        # add score:
        label['score'] = 100.0
        # create output:
        output_filename = os.path.join(
            output_dir, 'data', os.path.basename(input_filename)
        )
        label.to_csv(output_filename, sep=' ', header=False, index=False)

def get_arguments():
    """ 
    Get command-line arguments

    """
    # init parser:
    parser = argparse.ArgumentParser("Generate KITTI 3D Object Detection result from ground truth labels.")

    # add required and optional groups:
    required = parser.add_argument_group('Required')

    # add required:
    required.add_argument(
        "-i", dest="input", help="Input path of ground truth labels.",
        required=True, type=str
    )
    required.add_argument(
        "-o", dest="output", help="Output path of detection results.",
        required=True, type=str
    )

    # parse arguments:
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments:
    arguments = get_arguments()

    generate_detection_results(arguments.input, arguments.output)
```

The output from KITTI evaluation toolkit can be found here (click to follow the link) **[/workspace/assignments/06-deep-detection/doc/eval-on-ground-truth](here)**

---

### Generate Object Detection Results Using Point Pillars Prediction

In order to generate object detection results from point pillar predictions, use the adapted point pillars implementation by TuSimple (click to follow the link) **[/workspace/assignments/06-deep-detection/point-pillar/README.md](README.md)**

First, train the detectors for **car** and **pedestrian & cyclist**

```bash
# train car detector:
python ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.config --model_dir=/models/models/16/car
# train pedestrian & cyclist detector:
python ./pytorch/train.py train --config_path=./configs/pointpillars/ped_cycle/xyres_16.config --model_dir=/models/models/16/ped_cycle
```

Then get prediction output as follows:

```bash
# evaluate car detector:
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.config --model_dir=/models/models/16/car --measure_time=True --batch_size=4 --pickle_result=False
# evaluate pedestrian & cyclist detector:
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/ped_cycle/xyres_16.config --model_dir=/models/models/16/ped_cycle --measure_time=True --batch_size=4 --pickle_result=False
```

Finally, evaluate model output using KITTI evaluation toolkit:

```bash
# TBD
```

The output from KITTI evaluation toolkit can be found here (click to follow the link) **[/workspace/assignments/06-deep-detection/doc/eval-on-pointpillars-trained](here)**