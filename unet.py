__author__ = "Anil Appak"
__email__ = "ipekanilatalay@gmail.com"
__organization__ = "Tampere University"

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Channel Attention (SE) ----------
class SE(nn.Module):
    def __init__(self, C, r=8, bias_to_one=2.0):
        super().__init__()
        h = max(1, C // r)
        self.fc1 = nn.Linear(C, h)
        self.fc2 = nn.Linear(h, C)
        # start near identity so we don't throttle early
        nn.init.constant_(self.fc2.bias, bias_to_one)

    def forward(self, x):          # x: [B,C,H,W]
        s = x.mean(dim=(2, 3))     # [B,C]
        a = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))  # (0,1)
        gate = 0.5 + 0.5 * a       # ~[0.5,1.0], starts ~0.94
        return x * gate[:, :, None, None]


# ---------- Fourier Channel Attention (spectral attention) ----------
class FourierChannelAttention(nn.Module):
    """
    Scores each channel by its Fourier magnitude statistics and gates features.
    Cheap and works well with few wavelengths (e.g., 4).
    """
    def __init__(self, C, r=8, bias_to_one=2.0):
        super().__init__()
        h = max(1, C // r)
        self.mlp = nn.Sequential(
            nn.Linear(C, h), nn.ReLU(inplace=True),
            nn.Linear(h, C)
        )
        self.fca_ln = nn.LayerNorm(C, elementwise_affine=False)

        # near-identity start
        with torch.no_grad():
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=1.0)
                    nn.init.zeros_(m.bias)
            self.mlp[-1].bias.fill_(bias_to_one)

    def forward(self, x):                  # x: [B,C,H,W]
        X = torch.fft.rfft2(x, norm="backward")  # [B,C,H,Wf]
        mag = X.abs()
        fvec = mag.mean(dim=(2, 3))             # [B,C]
        #fvec = self.fca_ln(fvec)            # remove per-sample channel-scale bias, channels clearly have different contrast/texture across λ, 
                                            #which often means different HF energy/SNR. In that case LN (or a scale-invariant statistic) helps.
        print('fvec mean per λ:', fvec.mean(0))      # large spread? -> use LN
        a = torch.sigmoid(self.mlp(fvec))       # (0,1)
        a = a - a.mean(dim=1, keepdim=True)          # relative, zero-mean per sample
        gate = 1.0 + 0.3 * a                         # small ±30% modulation around 1
        x = x * gate[:, :, None, None]
        print('gate mean per λ:', (1+0.3*torch.sigmoid(self.mlp(fvec))).mean(0))  # always same λ on top?

        #gate = 0.5 + 0.5 * a                    # ~[0.5,1.0]
        return x #x * gate[:, :, None, None]


# ---------- building blocks ----------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, gn_groups=8):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(gn_groups, mid_channels), num_channels=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(gn_groups, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, gn_groups=8):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, gn_groups=gn_groups)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, gn_groups=8):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, gn_groups=gn_groups)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, gn_groups=gn_groups)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if shapes mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # fusion
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


# ---------- UNet ----------
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, gn_groups=8):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Per-channel normalization at the very front (learnable affine)
        self.in_norm = nn.InstanceNorm2d(n_channels, affine=True, eps=1e-5)

        # Identity-initialized spectral 1x1 mixing
        self.spectral_mix = nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=True)
        with torch.no_grad():
            self.spectral_mix.weight.zero_()
            for c in range(n_channels):
                self.spectral_mix.weight[c, c, 0, 0] = 1.0
            self.spectral_mix.bias.zero_()

        # Fourier Channel Attention (spectral attention)
        self.fca = FourierChannelAttention(n_channels, r=8, bias_to_one=2.0)

        # Encoder
        self.inc   = DoubleConv(n_channels, 32, gn_groups=gn_groups)
        self.down1 = Down(32,  64, gn_groups=gn_groups)
        self.down2 = Down(64,  128, gn_groups=gn_groups)
        factor = 2 if bilinear else 1
        self.down3 = Down(128, 256 // factor, gn_groups=gn_groups)

        # Decoder + SE after each fusion
        self.up2     = Up(256, 128 // factor, bilinear, gn_groups=gn_groups)
        self.se_dec2 = SE(128 // factor)
        self.up3     = Up(128, 64 // factor, bilinear, gn_groups=gn_groups)
        self.se_dec3 = SE(64 // factor)
        self.up4     = Up(64, 32, bilinear, gn_groups=gn_groups)
        self.se_dec4 = SE(32)

        self.outc = OutConv(32, n_classes)

        # If you want residual later:
        self.use_residual_head = True
        self.res_scale = 0.5  # only used if use_residual_head=True

        self.out_gain = nn.Parameter(torch.ones(1, n_classes, 1, 1))     # start ~1–2
        self.out_bias = nn.Parameter(torch.zeros(1, n_classes, 1, 1))    # start 0

    def forward(self, x):
        # front-end normalization and spectral attention
        x = self.in_norm(x)
        x_mixed = self.spectral_mix(x)
        x_mixed = self.fca(x_mixed)

        # encoder
        x1 = self.inc(x_mixed)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # decoder with SE after fusions
        y  = self.up2(x4, x3)
        #y  = self.se_dec2(y)
        y  = self.up3(y,  x2)
        #y  = self.se_dec3(y)
        y  = self.up4(y,  x1)
        #y  = self.se_dec4(y)

        o  = self.outc(y)

        # Stable head
        if not self.use_residual_head:
            out = torch.sigmoid(o)              # bounded, non-saturating early
        else:
            #out = (x_mixed + self.res_scale * o).clamp(0, 1)
            out_pre = x_mixed + self.res_scale * o                   # residual head
            out = torch.sigmoid(self.out_gain * out_pre + self.out_bias)  # <— no hard clamp

        return out
