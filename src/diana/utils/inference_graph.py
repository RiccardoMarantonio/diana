"""Capture the fixed-shape denoiser forward used by each reverse diffusion hop."""

import torch
from torch import nn


class InferenceGraph:
    """Keep random sampling outside capture and reuse one denoiser graph.

    The output aliases graph storage and must be consumed before the next call.
    A changed batch shape replaces the graph, bounding retained GPU memory.
    """

    def __init__(self, model: nn.Module, enabled: bool):
        self.model = model
        self.enabled = enabled and next(model.parameters()).device.type == "cuda"
        self.graph: torch.cuda.CUDAGraph | None = None
        self.x: torch.Tensor | None = None
        self.t: torch.Tensor | None = None
        self.output: torch.Tensor | None = None

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.model(x, t)
        if self.x is None or self.x.shape != x.shape:
            self.graph = None
            self.x, self.t = x.clone(), t.clone()
            try:
                stream = torch.cuda.Stream(device=x.device)
                stream.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.model(self.x, self.t)
                torch.cuda.current_stream(x.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.output = self.model(self.x, self.t)
                self.graph = graph
            except Exception as exc:  # noqa: BLE001 - capture may fail on unsupported kernels
                self.enabled = False
                self.graph = None
                self.x = self.t = self.output = None
                print(f"[inference graph] capture failed ({exc}); using eager inference")
                return self.model(x, t)
        assert self.graph is not None and self.x is not None and self.t is not None
        self.x.copy_(x)
        self.t.copy_(t)
        self.graph.replay()
        assert self.output is not None
        return self.output
