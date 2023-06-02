# DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks

This repository contains a reimplementation of the Automatic Photo Quality Enhancement using Deep Convolutional Networks project. The project aims to enhance the quality of photos using deep convolutional networks.

## Environment Setup
To set up the environment, please follow these steps:

```bash
pip install -r ../requirements.txt
```

## Training
To train the model, use the following command:

```bash
python train.py batch_size=<value> train_size=<value> lr=<value> epochs=<value> w_content=<value> w_color=<value> w_texture=<value> w_tv=<value> datadir=<path> vgg_pretrained=<path> eval_step=<value> model=<phone>
```

### Training Arguments
The following arguments can be passed to the `train.py` script:

- `batch_size=<value>`: Set the batch size for training. Default: `50`.
- `train_size=<value>`: Set the number of training samples. Default: `30000`.
- `lr=<value>`: Set the learning rate. Default: `5e-4`.
- `epochs=<value>`: Set the number of training epochs. Default: `20000`.
- `w_content=<value>`: Set the weight for the content loss. Default: `10`.
- `w_color=<value>`: Set the weight for the color loss. Default: `0.5`.
- `w_texture=<value>`: Set the weight for the texture loss. Default: `1`.
- `w_tv=<value>`: Set the weight for the total variation loss. Default: `2000`.
- `datadir=<path>`: Specify the directory containing the DPED dataset. Default: `dped/`.
- `vgg_pretrained=<path>`: Specify the path to the VGG-19 pretrained model weights. Default: `vgg_pretrained/imagenet-vgg-verydeep-19.mat`.
- `eval_step=<value>`: Set the evaluation step. Default: 1000.
- `model=<phone>`: Specify the camera model to train. Available options: `iphone`, `blackberry`, `sony`.

Example usage:
```bash
python train.py batch_size=32 train_size=30000 lr=5e-4 epochs=20000 w_content=10 w_color=0.5 w_texture=1 w_tv=2000 datadir=dped/ vgg_pretrained=vgg_pretrained/imagenet-vgg-verydeep-19.mat eval_step=1000 model=iphone
```

## Prediction
To predict and enhance the quality of photos, use the following command:

```bash
python predict.py model=<phone> datadir=<path> test_subset=<subset> iteration=<value> resolution=<value> use_gpu=<value>
```

### Prediction Arguments
The following arguments can be passed to the `predict.py` script:

- `model=<phone>`: Specify the camera model to use for prediction. Available options: `iphone`, `blackberry`, `sony`, `iphone_orig`, `blackberry_orig`, `sony_orig`.
- `datadir=<path>`: Specify the directory containing the test images. Default: `dped/`.
- `test_subset=<subset>`: Specify the subset of test images to predict. Default: `small`.
- `iteration=<value>`: Specify the iteration number of the trained model checkpoint to use for prediction. Default: `all`.
- `resolution=<value>`: Specify the resolution of the output enhanced images. Available options: `orig`, `

Example usage:
```bash
python predict.py model=iphone datadir=test_img/ test_subset=full iteration=18000 resolution=orig use_gpu=true
```

## Task Checklist:
1. Import Libraries       - done
2. Initial Setting        - done
3. Configure Data Loader  - done
4. Define Generator       - done
5. Define Discriminator   - done
6. Training               - done
7. Predict
8. Web (optional)
9. FINALIZATION (docstring, format, readme)

# Citation
```
@inproceedings{ignatov2017dslr,
  title={DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks},
  author={Ignatov, Andrey and Kobyshev, Nikolay and Timofte, Radu and Vanhoey, Kenneth and Van Gool, Luc},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={3277--3285},
  year={2017}
}
```
