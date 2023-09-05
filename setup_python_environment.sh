conda create -n attentd python=3.8 -y
# activate the virtual environment
conda activate attentd
conda install pytorch=1.11 torchvision=0.12 cudatoolkit=11.3 -c pytorch
pip install transformers==4.10.3

