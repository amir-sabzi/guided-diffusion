#!/bin/bash
cd ..
pip install --no-index --upgrade pip
pip install pytorch=1.7.0 --no-index
pip install torchvision=0.8.1 --no-index
pip install numpy=1.19.2 --no-index
pip install mpi4py --no-index
pip install matplotlib --no-index
pip install imageio --no-index
pip install matplotlib --no-index
pip install -e . --no-index
cd -