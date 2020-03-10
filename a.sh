#!/bin/sh
# run to reproduce link prediction experiments

# requirements: Ignore if already installed
pip install torch-cluster 
pip install torch-scatter==latest+cu101 torch-sparse==latest+cu101 torch-spline-conv==latest+cu101 -f https://s3.eu-central-1.amazonaws.com/pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric
pip install tensorboardX

python main.py --dataset brightkite
python main.py --dataset ppi --epoch_num 500