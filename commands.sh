cd ckpt
cd densepose
rm model_final_162be9.pkl
wget https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl
cd ..

cd humanparsing
rm parsing_atr.onnx
rm parsing_lip.onnx
wget https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx
wget https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx
cd ..

cd openpose
cd ckpts
rm body_pose_model.pth
wget https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth
cd ..
cd ..
cd ..

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

conda env create -f environment.yaml
conda activate idm

pip install huggingface_hub==0.25.1
