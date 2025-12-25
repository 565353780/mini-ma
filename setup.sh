cd ..
git clone https://github.com/565353780/camera-control.git
git clone https://github.com/565353780/non-rigid-icp.git

cd camera-control
./setup.sh

cd ../non-rigid-icp
./setup.sh

pip install numpy opencv_python einops kornia loguru yacs \
  tqdm yq jupyterlab matplotlib pytorch-lightning scipy \
  joblib iopath
