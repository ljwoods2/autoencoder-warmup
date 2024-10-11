# miniJSRT Autoencoder tasks

## Setup

Download data
```bash
wget http://imgcom.jsrt.or.jp/imgcom/wp-content/uploads/2019/07/autoencoder_img.zip .
unzip autoencover_imd.zip
```

Setup environment
```bash
mamba env create --name warmup python=3.11
pip install -r requirements.txt
module load cuda-12.6.1-gcc-12.1.0
module load gcc-12.1.0-gcc-11.2.0
anomalib install
```

## Classification of flipped images

Train the model
```bash
python classification.py train --data_root /scratch/ljwoods2/minijsrt/autoencoder_img
```

Run
```bash
python classification.py inference --data_root /scratch/ljwoods2/minijsrt/autoencoder_img \
    --model_path /home/ljwoods2/workspace/image-warmup-autoencoders/results/Patchcore/autoencoder_img/v0/weights/lightning/model.ckpt
```

## Unsupervised clustering