cd ..
git clone git@github.com:565353780/camera-control.git
git clone git@github.com:565353780/non-rigid-icp.git
git clone git@github.com:565353780/cage-deform.git

cd camera-control
./dev_setup.sh

cd ../non-rigid-icp
./dev_setup.sh

cd ../cage-deform
./dev_setup.sh

pip install numpy opencv_python einops kornia loguru yacs \
  tqdm yq jupyterlab matplotlib pytorch-lightning scipy \
  joblib iopath
