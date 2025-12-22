cd ..
git clone https://github.com/facebookresearch/pytorch3d.git

pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

pip install numpy opencv_python einops kornia loguru yacs \
  tqdm yq jupyterlab matplotlib pytorch-lightning scipy \
  joblib

cd pytorch3d
python setup.py install
