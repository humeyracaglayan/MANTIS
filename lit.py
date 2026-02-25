import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import gc

from unet import UNet
from optic_blackbox import OpticBlackBox
from torchmetrics.functional import structural_similarity_index_measure as ssim


def gradient_loss(pred, gt):
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_g = gt[..., 1:, :] - gt[..., :-1, :]
    dx_g = gt[..., :, 1:] - gt[..., :, :-1]
    return (dy_p - dy_g).abs().mean() + (dx_p - dx_g).abs().mean()


def lowfreq_mse(pred, gt, k=4):
    return F.mse_loss(F.avg_pool2d(pred, k, k), F.avg_pool2d(gt, k, k))


class DeepDoFLit(pl.LightningModule):
    def __init__(
        self,
        optic_blackbox_path: str,
        n_phi: int,
        noise_sigma: float,
        in_channels: int,
        out_channels: int,
        step_1: dict,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.n_phi = int(n_phi)
        self.noise_sigma = float(noise_sigma)
        self.step_1 = step_1

        if optic_blackbox_path is None:
            raise ValueError("optic_blackbox_path must be provided")

        self.optic = OpticBlackBox(optic_blackbox_path, n_phi=self.n_phi, run_on_cpu=True)
        self.deblurring_network = UNet(in_channels, out_channels, bilinear=False).float()

    def on_fit_start(self):
        # Move modules to the real device once Lightning placed the model.
        self.optic = self.optic.to(self.device).eval()
        self.deblurring_network = self.deblurring_network.to(self.device).train()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device).float()
        y = y.to(self.device).float()

        x_stack = x.repeat(self.n_phi, 1, 1, 1)
        y_stack = y.repeat(self.n_phi, 1, 1, 1)

        opt = self.optimizers()

        with torch.no_grad():
            optic_out = self.optic(x_stack).float()

        noise = torch.normal(0.0, self.noise_sigma, size=optic_out.shape, device=self.device)
        sensor_noisy = torch.relu(optic_out + noise)

        pred = self.deblurring_network(sensor_noisy)

        rmse_loss = torch.sqrt(F.mse_loss(pred, y_stack))
        l_grad = gradient_loss(pred, y_stack)
        l_lf = lowfreq_mse(pred, y_stack, k=4)

        pool = 4
        pred_ssim = F.avg_pool2d(pred, pool, pool)
        gt_ssim = F.avg_pool2d(y_stack, pool, pool)

        autocast_enabled = (self.device.type == "cuda")
        with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
            ssim_loss = 0.0
            for c in range(pred_ssim.shape[1]):
                ssim_loss = ssim_loss + (1.0 - ssim(pred_ssim[:, c:c+1], gt_ssim[:, c:c+1], data_range=1.0))
            ssim_loss = ssim_loss / pred_ssim.shape[1]

        loss = 0.65 * rmse_loss + 0.15 * ssim_loss + 0.10 * l_grad + 0.10 * l_lf

        opt.zero_grad()
        loss.backward()
        opt.step()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/rmse", rmse_loss, on_step=True, on_epoch=True)
        self.log("train/ssim", ssim_loss, on_step=True, on_epoch=True)
        self.log("train/grad", l_grad, on_step=True, on_epoch=True)
        self.log("train/lowfreq", l_lf, on_step=True, on_epoch=True)

        gc.collect()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device).float()
        y = y.to(self.device).float()

        x_stack = x.repeat(self.n_phi, 1, 1, 1)
        y_stack = y.repeat(self.n_phi, 1, 1, 1)

        with torch.no_grad():
            optic_out = self.optic(x_stack).float()
            noise = torch.normal(0.0, self.noise_sigma, size=optic_out.shape, device=self.device)
            sensor_noisy = torch.relu(optic_out + noise)
            pred = self.deblurring_network(sensor_noisy).clamp(0, 1)

        val_rmse = torch.sqrt(F.mse_loss(pred, y_stack))
        self.log("val/rmse", val_rmse, on_epoch=True, prog_bar=True)
        return val_rmse

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.deblurring_network.parameters(),
            lr=self.step_1["lr_digital"],
            weight_decay=1e-3,
        )