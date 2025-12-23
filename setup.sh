cd ..
git clone https://github.com/facebookresearch/pytorch3d.git
git clone https://github.com/565353780/camera-control.git

cd camera-control
./setup.sh

pip install ninja

pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

pip install numpy opencv_python einops kornia loguru yacs \
  tqdm yq jupyterlab matplotlib pytorch-lightning scipy \
  joblib trimesh open3d iopath

cd ../pytorch3d
python setup.py install
