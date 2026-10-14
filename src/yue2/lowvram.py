"""Low-VRAM weight placement for 8 GB GPUs (VitusSongs addition).

YuE2-3B is an AR-NAR mixture-of-transformers: about 4.3 GB of AR weights
(embedding, AR attention/MLP, LM head) and about 2.8 GB of NAR weights, 7.1 GB
in BF16 in total. The stages never need both halves at once, so only the half
required by the current stage is kept on the accelerator and the other half
waits in system RAM (the CPU copies are never rebuilt, see _move). Numerical results are identical to the all-on-GPU path;
only the extra host<->device copies cost time (a few seconds per swap).

  for_ar()           semantic / ABC token generation   -> AR weights on GPU
  for_nar_prefill()  AR prefix pass of a chunk         -> AR (no LM head) + small NAR parts
  for_nar_solve()    flow-matching ODE of a chunk      -> NAR weights on GPU
"""
from __future__ import annotations

import torch


def _same_device(a: torch.device, b: torch.device) -> bool:
    return a.type == b.type and (b.index is None or a.index is None or a.index == b.index)


class LowVRAMPlacement:
    def __init__(self, model, device):
        self.model, self.device = model, torch.device(device)
        backbone = model.model
        self.ar = [backbone.embed_tokens]
        self.nar = []
        for layer in backbone.layers:
            self.ar += [layer.input_layernorm, layer.self_attn, layer.post_attention_layernorm, layer.mlp]
            self.nar += [layer.nar_input_layernorm, layer.nar_self_attn,
                         layer.nar_pre_mlp_layernorm, layer.nar_mlp]
        self.head = [model.lm_head]
        self.shared = [m for m in (backbone.norm, getattr(backbone, "rotary_emb", None)) if m is not None]
        self.nar_small = [model.vae2llm, model.llm2vae, model.time_embedder, model.latent_pos_embed]
        self.cpu = torch.device("cpu")
        self._cpu = {}  # id(param) -> original CPU tensor

    def _move(self, modules, device):
        """Move modules to `device`. Weights are read-only at inference time, so the original CPU
        tensor of every parameter is kept: moving back to the CPU is then free (no copy, no new
        RAM allocation) and only host -> device transfers cost time."""
        device = torch.device(device)
        for module in modules:
            for param in module.parameters():
                if _same_device(param.device, device):
                    continue
                if device.type == "cpu":
                    cpu = self._cpu.get(id(param))
                    param.data = cpu if cpu is not None else param.data.to(device)
                else:
                    if param.device.type == "cpu":
                        self._cpu.setdefault(id(param), param.data)
                    param.data = param.data.to(device)
            for buf in module.buffers():
                if not _same_device(buf.device, device):
                    buf.data = buf.data.to(device)

    def _free(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def for_ar(self):
        self._move(self.nar + self.nar_small, self.cpu)
        self._free()
        self._move(self.ar + self.head + self.shared, self.device)

    def for_nar_prefill(self):
        self._move(self.nar + self.head, self.cpu)
        self._free()
        self._move(self.ar + self.shared + self.nar_small, self.device)

    def for_nar_solve(self):
        self._move(self.ar + self.head, self.cpu)
        self._free()
        self._move(self.nar + self.shared + self.nar_small, self.device)
