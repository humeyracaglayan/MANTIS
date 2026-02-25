import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

try:
    from piq import ssim as piq_ssim
    HAS_PIQ = True
except Exception:
    HAS_PIQ = False


def load_png_as_chw(path):
    x = np.asarray(Image.open(path), dtype=np.float32)
    if x.ndim == 2:
        x = x[..., None]
    if x.shape[2] == 4:
        x = x[:, :, :3]
    x = x / 255.0
    x = x / (x.max() + 1e-8)
    x = torch.from_numpy(x).permute(2, 0, 1).contiguous()  # [C,H,W]
    return x


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).clamp_min(1e-12)
    return float(20.0 * torch.log10(torch.tensor(1.0, device=a.device) / torch.sqrt(mse)))


def ssim2d(a, b):
    if not HAS_PIQ:
        return float("nan")
    a = a[None, None].double()
    b = b[None, None].double()
    dr = float(max(a.max().item(), b.max().item(), 1.0))
    return float(piq_ssim(a, b, data_range=dr).detach().cpu().item())


def save_grid(img4d, out_png, title, metrics=None):
    # img4d: [D,C,H,W]
    D, C, _, _ = img4d.shape
    fig, axes = plt.subplots(D, C, figsize=(C * 2.2, D * 2.2))
    if D == 1 and C == 1:
        axes = np.array([[axes]])
    elif D == 1:
        axes = axes.reshape(1, -1)
    elif C == 1:
        axes = axes.reshape(-1, 1)

    for d in range(D):
        for c in range(C):
            axes[d, c].imshow(img4d[d, c].detach().cpu().numpy(), vmin=0, vmax=1)
            axes[d, c].set_xticks([])
            axes[d, c].set_yticks([])
            if metrics is not None and (d, c) in metrics:
                axes[d, c].text(
                    0.03, 0.05, metrics[(d, c)],
                    transform=axes[d, c].transAxes,
                    fontsize=8,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.5),
                )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    # Paths
    test_png = "test_img.png"
    optic_pt = "optic_module.pt" # You can replace this with your own optic model if you have one.
    deblur_pt = "deblurring_module.pt" # You can replace this with your own deblurring model if you have one.
    out_dir = "predict_outputs"

    # Settings
    noise_sigma = 0.01
    D = 5  # number of depths to repeat

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    #######################################################
    # Load models
    #######################################################
    optic = torch.jit.load(optic_pt, map_location="cpu").eval()
    deblur = torch.jit.load(deblur_pt, map_location=device).eval()
    #######################################################

    #######################################################
    # Load and prepare input
    #######################################################

    x = np.asarray(Image.open('test_img.png'), dtype=np.float32) / 255.0 
    x = x / (x.max() + 1e-8)

        # Convert to PyTorch tensor and permute to [C, H, W]
    if x.ndim == 2:  # Grayscale
        x = torch.from_numpy(x).unsqueeze(0)
    else:  # RGB or multi-channel
        x = torch.from_numpy(x).permute(2, 0, 1)

    x_stack = x.unsqueeze(0).repeat(D, 1, 1, 1)      # [D,C,H,W]
    x_stack = x_stack.to(device).float().clamp(0, 1)
    #######################################################

    #######################################################
    # Run inference
    #######################################################
    with torch.no_grad():
        sensor = optic(x_stack.cpu())   
        if isinstance(sensor, (tuple, list)):
            sensor = sensor[0]
        sensor = sensor.float().to(device)

        noise = torch.normal(0.0, noise_sigma, size=sensor.shape, device=sensor.device)
        sensor_noisy = torch.relu(sensor + noise)

        out = deblur(sensor_noisy)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.float().clamp(0, 1)

    # Metrics
    metrics_sensor = {}
    metrics_out = {}
    rows = []

    for d in range(D):
        for c in range(x_stack.shape[1]):
            gt = x_stack[d, c].clamp(0, 1)
            s = sensor_noisy[d, c].clamp(0, 1)
            y = out[d, c].clamp(0, 1)

            ps_s = psnr(gt, s)
            ps_y = psnr(gt, y)
            ss_y = ssim2d(gt, y)

            metrics_sensor[(d, c)] = f"PSNR {ps_s:.2f}"
            if np.isnan(ss_y):
                metrics_out[(d, c)] = f"PSNR {ps_y:.2f}"
            else:
                metrics_out[(d, c)] = f"PSNR {ps_y:.2f}\nSSIM {ss_y:.3f}"

            rows.append({
                "depth": d,
                "channel": c,
                "psnr_sensor": ps_s,
                "psnr_output": ps_y,
                "ssim_output": ss_y,
            })

    #######################################################

    #######################################################
    # Save results
    #######################################################
    save_grid(x_stack, os.path.join(out_dir, "Groundtruth.png"), "Groundtruth")
    save_grid(sensor_noisy.clamp(0, 1), os.path.join(out_dir, "Sensor.png"), "Sensor", metrics_sensor)
    save_grid(out, os.path.join(out_dir, "Output.png"), "Output", metrics_out)

    # Save metrics
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Saved to", out_dir)
    if not HAS_PIQ:
        print("Install piq to compute SSIM")
    #######################################################


if __name__ == "__main__":
    main()