echo "****************** Installing pytorch ******************"
#conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
#server version=1.5.0 docker hub = pytorch/pytorch:1.5-cuda10.1-cudnn7-devel /bin/bash
echo ""
echo ""
echo "****************** Installing yaml ******************"
pip install PyYAML

echo ""
echo ""
echo "****************** Installing easydict ******************"
pip install easydict

echo ""
echo ""
echo "****************** Installing cython ******************"
pip install cython

echo ""
echo ""
echo "****************** Installing opencv-python ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing pandas ******************"
pip install pandas

echo ""
echo ""
echo "****************** Installing tqdm ******************"
conda install -y tqdm

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
pip install tb-nightly

echo ""
echo ""
echo "****************** Installing tikzplotlib ******************"
pip install tikzplotlib

echo ""
echo ""
echo "****************** Installing thop tool for FLOPs and Params computing ******************"
pip install thop==0.1.1.post2209072238

echo ""
echo ""
echo "****************** Installing colorama ******************"
pip install colorama

echo ""
echo ""
echo "****************** Installing lmdb ******************"
pip install lmdb

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom


echo ""
echo ""
echo "****************** Installing tensorboardX ******************"
pip install tensorboardX


echo ""
echo ""
echo "****************** Downgrade setuptools ******************"
pip install setuptools==59.5.0


echo ""
echo ""
echo "****************** Installing wandb ******************"
pip install wandb

echo ""
echo ""
echo "****************** Installing timm ******************"
pip install timm==0.6.7

echo ""
echo ""
apt install libgl1-mesa-glx
apt-get install ffmpeg libsm6 libxext6  -y

echo "****************** Installing munkres ******************"
pip install munkres==1.1.4
echo "****************** Installing shapely ******************"

pip install shapely
echo "****************** Installing vot challenge toolkit ******************"
pip install git+https://github.com/votchallenge/vot-toolkit-python
echo "****************** setting of workspace ******************"
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output

echo "****************** Installation complete! ******************"
