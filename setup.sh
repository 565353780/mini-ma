cd ..
git clone https://github.com/565353780/camera-control.git

cd camera-control
./setup.sh

pip install numpy opencv_python einops kornia loguru yacs \
  tqdm yq jupyterlab matplotlib pytorch-lightning scipy \
  joblib iopath
