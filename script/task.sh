#!/bin/bash
#SBATCH --job-name=test          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=5-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=<YourNetID>@princeton.edu
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#SBATCH --nodelist=selab2
#SBATCH -o/home/lpnquynh/desktop/dped-pytorch/script/test.out
#SBATCH -e/home/lpnquynh/desktop/dped-pytorch/script/test.err

module purge
module load anaconda3-2021.05-gcc-9.3.0-r6itwa7
# export PATH=/usr/local/cuda-11.5/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/

nvidia-smi
nvcc --version

conda init bash 
source activate cinnamon

## ENV ##
# pip install -r ../requirements.txt
# conda install -c conda-forge cudatoolkit=11.8.0
# python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
# mkdir -p $CONDA_PREFIX/etc/conda/activate.d
# echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# # Verify install:
# python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

## TRAIN ##
# python train_model.py batch_size=32 model=iphone dped_dir=dped/ vgg_dir=vgg_pretrained/imagenet-vgg-verydeep-19.mat

## PREDICT
# python test_model.py dped_dir=test_img/ iteration=18000 model=iphone test_subset=full resolution=orig use_gpu=true


## TEST
cd ..
# python train.py model=iphone train_size=10 datadir=dped/ vgg_pretrained=pretrained/imagenet-vgg-verydeep-19.mat
python test.py


conda deactivate



