import sys

import numpy as np
import time
import torch
from torch import nn
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import utils
from utils import parse_arg,dataset,metrics,models,preprocess,func
from utils.vgg import VGGNet
from tqdm import tqdm

# defining size of the training image patches

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

def run(
    batch_size,
    train_size,
    lr,
    epochs,
    w_content,
    w_color,
    w_texture,
    w_tv,
    datadir,
    vgg_pretrained,
    eval_step,
    phone,
):
    
    # default param
    num_workers=4

    # defining system architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed
    torch.manual_seed(0)

    # Load VGG model for content loss
    vgg_model = VGGNet(vgg_pretrained)
    vgg_model.to(device).eval()

    # Load dataset
    train_ds=dataset.DPEDTrainDataset(phone, datadir, train_size,is_train=True)
    # test_ds=dataset.DPEDTrainDataset(phone, datadir, train_size,is_train=False)

    # Dataloader
    train_dl= DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,collate_fn=train_ds.collate_fn)
    # test_dl= DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
   
    # Create the model
    generator = models.ResNet()
    generator=generator.to(device)
    discriminator=models.Adversarial()
    discriminator=discriminator.to(device)
    # Set optimizers
    gen_optimizer = torch.optim.AdamW(
        generator.parameters(), lr=lr, weight_decay=0.0001)
    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=lr, weight_decay=0.0001)
    
    # Training loop
    print('Start training')
    generator.train()
    discriminator.train()
    for i in range(epochs):
        train_loss_gen = 0.0
        train_acc_discrim = 0.0

        for batch in tqdm(train_dl):
            phone_images, dslr_images = batch['phone_image'].float().to(device), batch['dslr_image'].float().to(device)

            current_batch_size=min(batch_size,phone_images.shape[0])

            # Zero gradients
            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            # A. FORWARD
            # Reshape to fit into the net
            phone_images=phone_images.view(-1,PATCH_HEIGHT, PATCH_WIDTH,3).permute(0, 3, 1, 2) # (batch,3,100,100)
            dslr_images=dslr_images.view(-1,PATCH_HEIGHT, PATCH_WIDTH,3).permute(0, 3, 1, 2) # (batch,3,100,100)
            
            # GENERATOR
            enhanced = generator(phone_images)
            assert phone_images.shape == enhanced.shape,"image shape changed after generator"

            ## Transform both dslr and enhanced images to grayscale
            # Convert enhanced RGB images to grayscale
            enhanced_gray = F.rgb_to_grayscale(enhanced)

            # Convert DSLR RGB images to grayscale
            dslr_gray = F.rgb_to_grayscale(dslr_images)

            # Reshape the grayscale images
            enhanced_gray = enhanced_gray.view(-1, PATCH_HEIGHT * PATCH_WIDTH)
            dslr_gray = dslr_gray.view(-1, PATCH_HEIGHT * PATCH_WIDTH)

            # Randomly swap for discriminator
            chosen=torch.randint(0, 2, (current_batch_size,),dtype=torch.float32).unsqueeze(1).to(device) # 0: enhance, 1: dslr
            adversarial= enhanced_gray * (1 - chosen) + dslr_gray * chosen
            adversarial_images = adversarial.view(-1, 1, PATCH_HEIGHT, PATCH_WIDTH)

            # DISCRIMINATOR
            discrim_predictions = discriminator(adversarial_images)
            print("discrim result",discrim_predictions.shape)
            print(discrim_predictions)

            # B. LOSS
            # 1) Texture (adversarial) loss

            discrim_target = torch.cat([chosen, 1 - chosen], dim=1)
            loss_discrim = -torch.sum(discrim_target * torch.log(torch.clamp(discrim_predictions, 1e-10, 1.0)))
            loss_texture = -loss_discrim

            correct_predictions = (torch.argmax(discrim_predictions, dim=1)==torch.argmax(discrim_target, dim=1)).bool().int()
            discrim_accuracy = torch.mean(correct_predictions.float())

            print('loss_texture',loss_texture)
            print('correct_predictions',correct_predictions)
            print('discrim_accuracy',discrim_accuracy)

            # 2) Content loss
            CONTENT_LAYER = 'relu5_4'
            enhanced_vgg = vgg_model(vgg_model.preprocess(enhanced * 255))
            dslr_vgg = vgg_model(vgg_model.preprocess(dslr_images * 255))
            content_size = func.tensor_size(dslr_vgg[CONTENT_LAYER]) * current_batch_size
            mse=nn.MSELoss()
            loss_content = 2 * mse(enhanced_vgg[CONTENT_LAYER], dslr_vgg[CONTENT_LAYER]) / content_size

            print('loss_content',loss_content)
            # 3) Color loss
            enhanced_blur = preprocess.gaussian_blur(enhanced)
            dslr_blur = preprocess.gaussian_blur(dslr_images)
            loss_color = torch.sum(torch.pow(dslr_blur - enhanced_blur, 2)) / (2 * current_batch_size)

            print('loss_color',loss_color)

            # 4) Total Variation (TV) Loss
            batch_shape = (batch_size, 3, PATCH_HEIGHT, PATCH_WIDTH)
            tv_y_size = utils._tensor_size(enhanced[:, 1:, :, :])
            tv_x_size = utils._tensor_size(enhanced[:, :, 1:, :])
            y_tv = torch.nn.L2Loss()(enhanced[:, 1:, :, :] - enhanced[:, :batch_shape[1]-1, :, :])
            x_tv = torch.nn.L2Loss()(enhanced[:, :, 1:, :] - enhanced[:, :, :batch_shape[2]-1, :])
            loss_tv = 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size

            # batch_shape = (batch_size, 3, PATCH_WIDTH, PATCH_HEIGHT)
            # tv_y_size = torch.numel(enhanced[:, :, 1:, :])
            # tv_x_size = torch.numel(enhanced[:, :, :, 1:])
            # y_tv = torch.nn.MSELoss()(enhanced[:, :, 1:, :] - enhanced[:, :, :batch_shape[2] - 1, :])
            # x_tv = torch.nn.MSELoss()(enhanced[:, :, :, 1:] - enhanced[:, :, :, :batch_shape[3] - 1])
            # loss_tv = 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size

            print('loss_tv',loss_tv)

            # # Generator Loss
            # loss_generator = w_content * loss_content + w_texture * loss_texture + w_color * loss_color + w_tv * loss_tv

            # # PSNR Loss
            # enhanced_flat = enhanced.reshape(-1, PATCH_SIZE)
            # loss_mse = torch.sum(torch.pow(dslr_ - enhanced_flat, 2)) / (PATCH_SIZE * batch_size)
            # loss_psnr = 20 * torch.log10(1.0 / torch.sqrt(loss_mse))

            break # for debug

        break
    #         gen_loss = w_content * content_loss_value + w_color * color_loss + w_texture * texture_loss + w_tv * tv_loss

    #         # Update generator
    #         gen_loss.backward()
    #         gen_optimizer.step()

    #         # Update discriminator
    #         disc_loss = torch.mean(torch.pow(discriminator(dslr_images) - 1, 2)) + \
    #                     torch.mean(torch.pow(discriminator(enhanced.detach()), 2))
    #         disc_loss.backward()
    #         disc_optimizer.step()

    #         # Accumulate losses and accuracy
    #         train_loss_gen += gen_loss.item()
    #         train_acc_discrim += torch.mean(torch.pow(discriminator(dslr_images) - 1, 2)).item() + \
    #                             torch.mean(torch.pow(discriminator(enhanced.detach()), 2)).item()

    #     # Print training losses
    #     if (i + 1) % eval_step == 0:
    #         train_loss_gen /= len(train_dataloader)
    #         train_acc_discrim /= len(train_dataloader)

    #         print(f"Iteration: {i+1}/{num_train_iters}, Gen Loss: {train_loss_gen:.4f}, Discrim Loss: {train_acc_discrim:.4f}")

    #     # Evaluate on test dataset
    #     if (i + 1) % eval_step == 0:
    #         generator.eval()
    #         test_loss_gen = 0.0
    #         test_psnr = 0.0
    #         test_ssim = 0.0
    #         num_batches = 0

    #         for batch in test_dataloader:
    #             phone_images, dslr_images = batch['phone'].to(device), batch['dslr'].to(device)
    #             enhanced = generator(phone_images)

    #             test_loss_gen += content_loss(vgg_model(enhanced), vgg_model(dslr_images)).item()
    #             test_psnr += compute_psnr(enhanced, dslr_images)
    #             test_ssim += SSIMLoss()(enhanced, dslr_images)

    #             num_batches += 1

    #         test_loss_gen /= num_batches
    #         test_psnr /= num_batches
    #         test_ssim /= num_batches

    #         print(f"Test Gen Loss: {test_loss_gen:.4f}, PSNR: {test_psnr:.2f}, SSIM: {test_ssim:.4f}")

    # # Save the trained generator model
    # torch.save(generator.state_dict(), "trained_generator.pth")
if __name__ == "__main__":
    param=parse_arg.train_args(sys.argv)
    run(**param)
