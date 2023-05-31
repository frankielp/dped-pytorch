import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io

IMAGE_MEAN = np.array([123.68, 116.779, 103.939])

class VGGNet(nn.Module):
    def __init__(self, path_to_vgg_net):
        super(VGGNet, self).__init__()
        self.layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )

        # Load pre-trained weights
        data = scipy.io.loadmat(path_to_vgg_net)
        weights = data['layers'][0]

        self.net = nn.ModuleDict()
        for i, name in enumerate(self.layers):
            layer_type = name[:4]
            if layer_type == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (3, 2, 0, 1))
                bias = bias.reshape(-1)
                conv_layer = self.conv_layer(kernels, bias)
                self.net[name] = conv_layer
            
            elif layer_type == 'relu':
                relu_layer = nn.ReLU()
                self.net[name] = relu_layer
              
            elif layer_type == 'pool':
                pool_layer = self.pool_layer()
                self.net[name] = pool_layer
               

    def forward(self, input_image):
        net_output = {}
        current = input_image
        for name, layer in self.net.items():
            current = layer(current)
            net_output[name] = current
        return net_output

    def conv_layer(self, weights, bias):
        conv = nn.Conv2d(weights.shape[1], weights.shape[0], kernel_size=weights.shape[2:], stride=1, padding=1)
        conv.weight.data = torch.from_numpy(weights)
        conv.bias.data = torch.from_numpy(bias)
        return conv

    def pool_layer(self):
        pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        return pool

def preprocess(image):
    return image - torch.tensor(IMAGE_MEAN)

if __name__=='__main__':
    print('Testing vgg.py')
    path_to_vgg_net = 'pretrained/imagenet-vgg-verydeep-19.mat'
    input_image = torch.randn(1, 3, 224, 224)
    vgg_net = VGGNet(path_to_vgg_net)
    net_output = vgg_net(input_image)
    print('vgg.py: Done')
