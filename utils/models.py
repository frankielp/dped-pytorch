import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.generator = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh(),
        )
        # Apply Xavier uniform weight initialization to the generator
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input_image):
        enhanced = self.generator(input_image)
        return enhanced


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.InstanceNorm2d(out_channels)

        # Apply Xavier uniform weight initialization to the convolutional layers
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out


class Adversarial(nn.Module):
    def __init__(self):
        super(Adversarial, self).__init__()

        self.discriminator = nn.Sequential(
            ConvLayer(1, 48, kernel_size=11, stride=4, batch_norm=False),
            ConvLayer(48, 128, kernel_size=5, stride=2),
            ConvLayer(128, 192, kernel_size=3, stride=1),
            ConvLayer(192, 192, kernel_size=3, stride=1),
            ConvLayer(192, 128, kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2),
            nn.Softmax(dim=1),
        )
        # Apply Xavier uniform weight initialization to the discriminator
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, image_):
        out = self.discriminator(image_)
        out_view = out.view(out.size(0), -1)
        adv_out = self.fc(out_view.clone())
        return adv_out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, batch_norm=True):
        super(ConvLayer, self).__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    print("Testing utils/models.py")
    torch.manual_seed(0)
    import numpy as np

    np.random.seed(0)

    PATCH_WIDTH = 100
    PATCH_HEIGHT = 100
    PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3
    # Create instances of the models
    resnet = ResNet()
    adversarial = Adversarial()

    batch = 50
    # Define the placeholders for training data
    phone_ = torch.Tensor()
    phone_ = phone_.resize_(batch, PATCH_SIZE).float()
    phone_image = phone_.view(-1, 3, PATCH_HEIGHT, PATCH_WIDTH)

    dslr_ = torch.Tensor()
    dslr_ = dslr_.resize_(batch, PATCH_SIZE).float()
    dslr_image = dslr_.view(-1, 3, PATCH_HEIGHT, PATCH_WIDTH)

    # Get processed enhanced image
    enhanced = resnet(phone_image)

    adv_ = torch.Tensor()
    adv_ = adv_.resize_(batch * 3, 1).float()

    # Transform both DSLR and enhanced images to grayscale
    enhanced_gray = enhanced.view(-1, PATCH_HEIGHT * PATCH_WIDTH)
    dslr_gray = dslr_image.view(-1, PATCH_HEIGHT * PATCH_WIDTH)

    print(enhanced.shape, enhanced_gray.shape, adv_.shape)
    # Push randomly the enhanced or DSLR image to an adversarial CNN-discriminator
    adversarial_ = enhanced_gray * (1 - adv_) + dslr_gray * adv_
    adversarial_image = adversarial_.view(-1, 1, PATCH_HEIGHT, PATCH_WIDTH)

    discrim_predictions = adversarial(adversarial_image)

    # Print the network architecture
    print("enhanced image\n", enhanced)
    print("adversarial image\n", adversarial_image)
    print("discrim predictions\n", discrim_predictions)
    print("the networks architectures\n")
    print(resnet)
    print(adversarial)
    print("models.py: Done")
