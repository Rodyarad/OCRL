import torch
from torch import nn


class SlotContrast_Module(nn.Module):
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        super().__init__()
        self.num_slots = ocr_config.num_slots
        self.rep_dim = ocr_config.slot_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Slot extraction is done in env wrapper. OCR module must only pass slots through.
        if obs.ndim == 3 and obs.shape[-2] == self.num_slots and obs.shape[-1] == self.rep_dim:
            return obs
        raise ValueError(
            "SlotContrast_Module expects slot observations of shape "
            f"(B, {self.num_slots}, {self.rep_dim}), got {tuple(obs.shape)}."
        )

    def get_loss(self, obs: torch.Tensor, with_rep=False) -> dict:
        if with_rep:
            return {}, self(obs)
        return {}

    def get_samples(self, obs: torch.Tensor) -> dict:
        return {}
