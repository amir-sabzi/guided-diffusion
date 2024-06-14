#!/bin/bash
cd ..
pip install --no-index --upgrade pip
pip install torch --no-index
pip install torchvision --no-index
pip install numpy --no-index
pip install matplotlib --no-index
pip install imageio --no-index
pip install matplotlib --no-index
pip install -e . --no-index
cd -