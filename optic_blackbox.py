__author__ = "Anil Appak"
__email__ = "ipekanilatalay@gmail.com"
__organization__ = "Tampere University"

import torch
import torch.nn as nn

class OpticBlackBox(nn.Module):
    def __init__(self, optic_pt_path: str, n_phi: int, run_on_cpu: bool = True):
        super().__init__()
        self.n_phi = int(n_phi)
        self.run_on_cpu = bool(run_on_cpu)

        if self.run_on_cpu:
            self.optic = torch.jit.load(optic_pt_path, map_location="cpu").eval()

            for name, buf in self.optic.named_buffers():
                if buf.device.type != "cpu":
                    raise RuntimeError(f"Optic buffer not on CPU: {name} on {buf.device}")
            for name, p in self.optic.named_parameters():
                if p.device.type != "cpu":
                    raise RuntimeError(f"Optic parameter not on CPU: {name} on {p.device}")

        else:
            self.optic = torch.jit.load(optic_pt_path).eval()

        for p in self.optic.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x_stack: torch.Tensor) -> torch.Tensor:
        if self.run_on_cpu:
            x_cpu = x_stack.detach().to("cpu").float()
            y = self.optic(x_cpu)
            if isinstance(y, (tuple, list)):
                y = y[0]
            return y.to(x_stack.device).float()

        if next(self.optic.parameters(), None) is not None:
            optic_device = next(self.optic.parameters()).device
        else:
            first_buf = next(self.optic.buffers(), None)
            optic_device = first_buf.device if first_buf is not None else x_stack.device

        if optic_device != x_stack.device:
            self.optic = self.optic.to(x_stack.device)

        y = self.optic(x_stack.float())
        if isinstance(y, (tuple, list)):
            y = y[0]
        return y.float()