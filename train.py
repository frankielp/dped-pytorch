import sys
import time
import os

import cv2
import imageio
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import dataset, func, models, parse_arg, preprocess
from utils.metrics import MultiScaleSSIM
from utils.vgg import VGGNet

# defining size of the training image patches

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

torch.autograd.set_detect_anomaly(True)


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
    num_workers = 4

    # defining system architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output folder
    output_path = 'models'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    folders = os.listdir(output_path)
    new_id = 0
    if len(folders) > 0:
        for folder in folders:
            if not folder.startswith('exp_'):
                continue
            new_id = max(new_id, int(folder.split('exp_')[-1]))
        new_id += 1
    output_path = os.path.join(output_path, f'exp_{new_id}')
    os.makedirs(output_path)
    weights_path = os.path.join(output_path, 'weights')
    os.mkdir(weights_path)
    visualize_path=os.path.join(output_path, 'visualize')
    os.mkdir(visualize_path)
    output_path=output_path+'/'
    visualize_path=visualize_path+'/'
    weights_path=weights_path+'/'

    # Set random seed
    torch.manual_seed(0)

    # Load VGG model for content loss
    vgg_model = VGGNet(vgg_pretrained)
    vgg_model.to(device).eval()

    # Load dataset
    train_ds = dataset.DPEDTrainDataset(phone, datadir, train_size, is_train=True)
    test_ds = dataset.DPEDTrainDataset(phone, datadir, train_size, is_train=False)

    # Dataloader
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=train_ds.collate_fn,
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Create the model
    generator = models.ResNet()
    generator = generator.to(device)
    discriminator = models.Adversarial()
    discriminator = discriminator.to(device)
    # Set optimizers
    gen_optimizer = torch.optim.AdamW(
        generator.parameters(), lr=lr, weight_decay=0.0001
    )
    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=lr, weight_decay=0.0001
    )

    # Training loop
    print("Start training")
    total_run = time.time()

    # Tracking loss
    # Training
    plot_train_loss_gen = []
    plot_train_loss_discrim = []
    plot_train_acc_discrim = []
    # Eval
    plot_test_loss_gen = []
    plot_test_loss_discrim = []
    plot_test_psnr = []
    plot_test_ssim = []
    plot_test_content = []
    plot_test_color = []
    plot_test_texture = []
    plot_test_tv = []
    plot_test_acc_discrim = []
    
    # Save log
    # To Overwrite
    logs = open(output_path + phone + ".txt", "w+")
    logs.close()

    for i in range(epochs):
        train_loss_gen = 0.0
        train_loss_discrim = 0.0
        train_acc_discrim = 0.0
        generator.train()
        discriminator.train()
        train_progress_bar = tqdm(enumerate(train_dl), total=len(train_dl))
        start_time = time.time()

        for _, batch in train_progress_bar:
            phone_images, dslr_images = batch["phone_image"].float().to(device), batch[
                "dslr_image"
            ].float().to(device)

            current_batch_size = min(batch_size, phone_images.shape[0])

            # A. FORWARD
            # Reshape to fit into the net
            phone_images = (
                phone_images.view(-1, PATCH_HEIGHT, PATCH_WIDTH, 3)
                .permute(0, 3, 1, 2)
                
            )  # (batch,3,100,100)
            dslr_images = (
                dslr_images.view(-1, PATCH_HEIGHT, PATCH_WIDTH, 3)
                .permute(0, 3, 1, 2)
                
            )  # (batch,3,100,100)

            # GENERATOR
            enhanced = generator(phone_images)
            assert (
                phone_images.shape == enhanced.shape
            ), "image shape changed after generator"

            ## Transform both dslr and enhanced images to grayscale
            # Convert enhanced RGB images to grayscale
            enhanced_gray = F.rgb_to_grayscale(enhanced)

            # Convert DSLR RGB images to grayscale
            dslr_gray = F.rgb_to_grayscale(dslr_images)

            # Reshape the grayscale images
            enhanced_gray = enhanced_gray.view(-1, PATCH_HEIGHT * PATCH_WIDTH)
            dslr_gray = dslr_gray.view(-1, PATCH_HEIGHT * PATCH_WIDTH)
            # Randomly swap for discriminator
            chosen = (
                torch.randint(0, 2, (current_batch_size,), dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )  # 0: enhance, 1: dslr
            adversarial = enhanced_gray * (1 - chosen) + dslr_gray * chosen
            adversarial_images = adversarial.view(-1, 1, PATCH_HEIGHT, PATCH_WIDTH)

            # DISCRIMINATOR
            discrim_predictions = discriminator(adversarial_images.detach())

            # B. LOSS
            # 1) Texture (adversarial) loss

            discrim_target = torch.cat([chosen, 1 - chosen], dim=1)
            loss_discrim = -torch.sum(discrim_target * torch.log(torch.clamp(discrim_predictions, 1e-10, 1.0)))
            loss_texture = -loss_discrim.detach()

            correct_predictions = torch.eq(
                torch.argmax(discrim_predictions, dim=1),
                torch.argmax(discrim_target, dim=1),
            ).float()
            discrim_accuracy = torch.mean(correct_predictions)

            # 2) Content loss
            CONTENT_LAYER = "relu5_4"
            enhanced_vgg = vgg_model(vgg_model.preprocess(enhanced * 255))
            dslr_vgg = vgg_model(vgg_model.preprocess(dslr_images * 255))
            content_size = (
                func.tensor_size(dslr_vgg[CONTENT_LAYER]) * current_batch_size
            )
            mse = nn.MSELoss()
            loss_content = (
                2
                * mse(enhanced_vgg[CONTENT_LAYER], dslr_vgg[CONTENT_LAYER])
                / content_size
            )

            # 3) Color loss
            enhanced_blur = preprocess.blur(enhanced)
            dslr_blur = preprocess.blur(dslr_images)
            loss_color = torch.sum(torch.pow(dslr_blur - enhanced_blur, 2)) / (
                2 * current_batch_size
            )

            # 4) Total Variation (TV) Loss
            batch_shape = (current_batch_size, 3, PATCH_HEIGHT, PATCH_WIDTH)

            tv_y_size = func.tensor_size(enhanced[:, :, :, 1:])
            tv_x_size = func.tensor_size(enhanced[:, :, 1:, :])
            mse = nn.MSELoss()
            y_tv = mse(enhanced[:, :, :, 1:], enhanced[:, :, :, : batch_shape[3] - 1])
            x_tv = mse(enhanced[:, :, 1:, :], enhanced[:, :, : batch_shape[2] - 1, :])
            loss_tv = 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / current_batch_size

            # 5) Generator Loss
            loss_generator = (
                w_content * loss_content
                + w_color * loss_color
                + w_tv * loss_tv
                + w_texture * loss_texture
            )
            # 6) PSNR Loss
            enhanced_flat = enhanced.reshape(-1, PATCH_SIZE)
            dslr_flat = dslr_images.reshape(-1, PATCH_SIZE)
            loss_mse = torch.sum(torch.pow(dslr_flat - enhanced_flat, 2)) / (
                PATCH_SIZE * current_batch_size
            )
            loss_psnr = 20 * torch.log10(1.0 / torch.sqrt(loss_mse))

            # C. BACKWARD

            # Update generator
            gen_optimizer.zero_grad()
            loss_generator.backward()
            gen_optimizer.step()

            # Update discriminator
            disc_optimizer.zero_grad()
            loss_discrim.backward()
            disc_optimizer.step()

            

            # Accumulate losses and accuracy
            train_loss_gen += loss_generator.item()
            train_loss_discrim += loss_discrim.item()
            train_acc_discrim += discrim_accuracy.item()
            # train_progress_bar.set_description(
            #     f"Epochs: {i+1}/{epochs}\n Train Loss - Generator: {train_loss_gen:.4f}, Discriminator: {train_loss_discrim:.4f} | Elapsed: {time.time() - start_time:.4f}"
            # )

        train_loss_gen /= len(train_dl)
        train_loss_discrim /= len(train_dl)
        train_acc_discrim /= len(train_dl)
        print(f"Epochs: {i+1}/{epochs}\n Train Loss - Generator: {train_loss_gen:.4f}, Discriminator: {train_loss_discrim:.4f} | Elapsed: {time.time() - start_time:.4f}")

        plot_train_loss_gen.append(train_loss_gen)
        plot_train_loss_discrim.append(train_loss_discrim)
        plot_train_acc_discrim.append(train_acc_discrim)

        # Evaluate on test dataset
        if (i + 1) % eval_step == 0:
            start_time = time.time()
            print("Start evaluating")
            generator.eval()
            test_loss_gen = 0.0
            test_psnr = 0.0
            test_ssim = 0.0
            test_content = 0.0
            test_color = 0.0
            test_texture = 0.0
            test_tv = 0.0
            test_acc_discrim = 0.0
            test_loss_discrim = 0.0
            test_progress_bar = tqdm(enumerate(test_dl), total=len(test_dl))

            for _, batch in test_progress_bar:
                with torch.no_grad():
                    eval_phone_images, eval_dslr_images = batch[
                        "phone_image"
                    ].float().to(device), batch["dslr_image"].float().to(device)

                    current_batch_size = min(batch_size, eval_phone_images.shape[0])

                    # A. FORWARD
                    # Reshape to fit into the net
                    eval_phone_images = (
                        eval_phone_images.view(-1, PATCH_HEIGHT, PATCH_WIDTH, 3)
                        .permute(0, 3, 1, 2)
                        
                    )  # (batch,3,100,100)
                    eval_dslr_images = (
                        eval_dslr_images.view(-1, PATCH_HEIGHT, PATCH_WIDTH, 3)
                        .permute(0, 3, 1, 2)
                        
                    )  # (batch,3,100,100)

                    # GENERATOR
                    eval_enhanced = generator(eval_phone_images)
                    assert (
                        eval_phone_images.shape == eval_enhanced.shape
                    ), "image shape changed after generator"

                    ## Transform both dslr and enhanced images to grayscale
                    # Convert enhanced RGB images to grayscale
                    eval_enhanced_gray = F.rgb_to_grayscale(eval_enhanced)

                    # Convert DSLR RGB images to grayscale
                    eval_dslr_gray = F.rgb_to_grayscale(eval_dslr_images)

                    # Reshape the grayscale images
                    eval_enhanced_gray = eval_enhanced_gray.view(
                        -1, PATCH_HEIGHT * PATCH_WIDTH
                    )
                    eval_dslr_gray = eval_dslr_gray.view(
                        -1, PATCH_HEIGHT * PATCH_WIDTH
                    )

                    # Randomly swap for discriminator
                    eval_chosen = (
                        torch.randint(0, 2, (current_batch_size,), dtype=torch.float32)
                        .unsqueeze(1)
                        .to(device)
                    )  # 0: enhance, 1: dslr
                    eval_adversarial = (
                        eval_enhanced_gray * (1 - eval_chosen)
                        + eval_dslr_gray * eval_chosen
                    )
                    eval_adversarial_images = eval_adversarial.view(
                        -1, 1, PATCH_HEIGHT, PATCH_WIDTH
                    )

                    # DISCRIMINATOR
                    eval_discrim_predictions = discriminator(
                        eval_adversarial_images.detach()
                    )

                    # B. LOSS
                    # 1) Texture (adversarial) loss

                    eval_discrim_target = torch.cat(
                        [eval_chosen, 1 - eval_chosen], dim=1
                    )
                    eval_loss_discrim = -torch.sum(
                        eval_discrim_target * torch.log(eval_discrim_predictions)
                    )
                    eval_loss_texture = -eval_loss_discrim.detach()

                    eval_correct_predictions = torch.eq(
                        torch.argmax(eval_discrim_predictions, dim=1),
                        torch.argmax(eval_discrim_target, dim=1),
                    ).float()
                    eval_discrim_accuracy = torch.mean(eval_correct_predictions)

                    # 2) Content loss
                    CONTENT_LAYER = "relu5_4"
                    eval_enhanced_vgg = vgg_model(
                        vgg_model.preprocess(eval_enhanced * 255)
                    )
                    eval_dslr_vgg = vgg_model(
                        vgg_model.preprocess(eval_dslr_images * 255)
                    )
                    eval_content_size = (
                        func.tensor_size(eval_dslr_vgg[CONTENT_LAYER])
                        * current_batch_size
                    )
                    mse = nn.MSELoss()
                    eval_loss_content = (
                        2
                        * mse(
                            eval_enhanced_vgg[CONTENT_LAYER],
                            eval_dslr_vgg[CONTENT_LAYER],
                        )
                        / eval_content_size
                    )

                    # 3) Color loss
                    eval_enhanced_blur = preprocess.blur(eval_enhanced)
                    eval_dslr_blur = preprocess.blur(eval_dslr_images)
                    eval_loss_color = torch.sum(
                        torch.pow(eval_dslr_blur - eval_enhanced_blur, 2)
                    ) / (2 * current_batch_size)

                    # 4) Total Variation (TV) Loss
                    batch_shape = (current_batch_size, 3, PATCH_HEIGHT, PATCH_WIDTH)

                    eval_tv_y_size = func.tensor_size(eval_enhanced[:, :, :, 1:])
                    eval_tv_x_size = func.tensor_size(eval_enhanced[:, :, 1:, :])
                    mse = nn.MSELoss()
                    eval_y_tv = mse(
                        eval_enhanced[:, :, :, 1:],
                        eval_enhanced[:, :, :, : batch_shape[3] - 1],
                    )
                    eval_x_tv = mse(
                        eval_enhanced[:, :, 1:, :],
                        eval_enhanced[:, :, : batch_shape[2] - 1, :],
                    )
                    eval_loss_tv = (
                        2
                        * (eval_x_tv / eval_tv_x_size + eval_y_tv / eval_tv_y_size)
                        / current_batch_size
                    )

                    # 5) Generator Loss
                    eval_loss_generator = (
                        w_content * eval_loss_content
                        + w_color * eval_loss_color
                        + w_tv * eval_loss_tv
                        + w_texture * eval_loss_texture
                    )
                    # 6) PSNR Loss
                    eval_enhanced_flat = eval_enhanced.reshape(-1, PATCH_SIZE)
                    eval_dslr_flat = eval_dslr_images.reshape(-1, PATCH_SIZE)
                    eval_loss_mse = torch.sum(
                        torch.pow(eval_dslr_flat - eval_enhanced_flat, 2)
                    ) / (PATCH_SIZE * current_batch_size)
                    eval_loss_psnr = 20 * torch.log10(1.0 / torch.sqrt(eval_loss_mse))

                    # CROP FOR VISUALIZE
                    random_indices = np.random.randint(0, current_batch_size, 5)

                    eval_enhanced_crop = eval_enhanced[random_indices, :]
                    eval_phone_images_crop = eval_phone_images[random_indices, :]

                    # EVAL SSIM
                    ssim_enhanced_images = eval_enhanced.permute(
                        0, 2, 3, 1
                    )  # (batch,100,100,3)
                    ssim_dslr_images = eval_dslr_images.permute(
                        0, 2, 3, 1
                    )  # (batch,100,100,3)
                    eval_ssim = MultiScaleSSIM(
                        ssim_dslr_images * 255, ssim_enhanced_images * 255
                    ) / current_batch_size

                    test_loss_gen += eval_loss_generator.item()
                    test_psnr += eval_loss_psnr.item()
                    test_ssim += eval_ssim
                    test_content += eval_loss_content.item()
                    test_color += eval_loss_color.item()
                    test_texture += eval_loss_texture.item()
                    test_tv += eval_loss_tv.item()
                    test_acc_discrim += eval_discrim_accuracy.item()
                    test_loss_discrim += eval_loss_discrim.item()

                    # test_progress_bar.set_description(
                    #     "Eval losses - generator (total): %.4g | content: %.4g, color: %.4g, texture: %.4g, tv: %.4g | psnr: %.4g, ms-ssim: %.4g | Elapsed:  %.4g\n "
                    #     % (
                    #         eval_loss_generator,
                    #         eval_loss_content,
                    #         eval_loss_color,
                    #         eval_loss_texture,
                    #         eval_loss_tv,
                    #         eval_loss_psnr,
                    #         eval_ssim,
                    #         (time.time() - start_time),
                    #     )
                    # )

            num_batches = len(test_dl)

            test_loss_gen /= num_batches
            test_psnr /= num_batches
            test_ssim /= num_batches
            test_content /= num_batches
            test_color /= num_batches
            test_texture /= num_batches
            test_tv /= num_batches
            test_acc_discrim /= num_batches
            test_loss_discrim /= num_batches

            plot_test_loss_gen.append(test_loss_gen)
            plot_test_loss_discrim.append(test_loss_discrim)
            plot_test_psnr.append(test_psnr)
            plot_test_ssim.append(test_ssim)
            plot_test_content.append(test_content)
            plot_test_color.append(test_color)
            plot_test_texture.append(test_texture)
            plot_test_tv.append(test_tv)
            plot_test_acc_discrim.append(test_acc_discrim)

            # Save plot
            func.plot_losses(
                plot_train_loss_gen,
                plot_train_loss_discrim,
                plot_train_acc_discrim,
                plot_test_loss_gen,
                plot_test_loss_discrim,
                plot_test_psnr,
                plot_test_ssim,
                plot_test_content,
                plot_test_color,
                plot_test_texture,
                plot_test_tv,
                plot_test_acc_discrim,
                f"{output_path}model_plot.png",
            )

            logs_disc = (
                "Epoch %d, %s | discriminator accuracy | train: %.4g, test: %.4g"
                % (
                    i,
                    phone,
                    train_acc_discrim,
                    test_acc_discrim,
                )
            )

            logs_gen = (
                "generator losses | train: %.4g, test: %.4g | content: %.4g, color: %.4g, texture: %.4g, tv: %.4g | psnr: %.4g, ms-ssim: %.4g\n"
                % (
                    train_loss_gen,
                    test_loss_gen,
                    test_content,
                    test_color,
                    test_texture,
                    test_tv,
                    test_psnr,
                    test_ssim,
                )
            )

            print(logs_disc)
            print(logs_gen)
            # Save log
            # To write
            logs = open(output_path + phone + ".txt", "a")
            logs.write(logs_disc)
            logs.write("\n")
            logs.write(logs_gen)
            logs.write("\n")
            logs.close()
            # Save visualization
            idx = 0
            for crop in eval_enhanced_crop:
                before_after = np.hstack(
                    (
                        eval_phone_images_crop[idx].permute(1, 2, 0).cpu().numpy(),
                        crop.permute(1, 2, 0).cpu().numpy(),
                    )
                )  # [ 3, 100, 100]->[100, 100,3]
                imageio.imwrite(visualize_path
                    + str(phone)
                    + "_"
                    + str(idx)
                    + "_iteration_"
                    + str(i)
                    + ".jpg",
                    cv2.normalize(
                        before_after,
                        None,
                        alpha=0,
                        beta=255,
                        norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U,
                    ).astype(np.uint8),
                )
                idx += 1

            # Save the trained generator model
            torch.save(generator.state_dict(), f"{weights_path}/generator_epoches_{i}.pth")
    print(f"Complete in {time.time()-total_run}")


if __name__ == "__main__":
    param = parse_arg.train_args(sys.argv)
    run(**param)
