import pytest
import torch

from diana.train import build_ddpm
from diana.utils.inference_graph import InferenceGraph
from tests.test_train import _cfg


def test_cpu_passthrough_matches_eager_reconstruction():
    model = build_ddpm(_cfg()).eval()
    graph = InferenceGraph(model.unet, enabled=True)
    x = torch.randn(2, 3, 16, 16)
    torch.manual_seed(4)
    expected = model.reconstruct(x, num_steps=8, t_start=8)
    torch.manual_seed(4)
    actual = model.reconstruct(x, num_steps=8, t_start=8, denoiser=graph)
    assert torch.equal(actual, expected)
    assert not graph.enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_replay_matches_eager_and_recaptures_tail():
    model = build_ddpm(_cfg()).cuda().eval()
    graph = InferenceGraph(model.unet, enabled=True)
    for batch in (4, 4, 1):
        x = torch.randn(batch, 3, 16, 16, device="cuda")
        t = torch.full((batch,), 8, device="cuda", dtype=torch.long)
        with torch.no_grad():
            expected = model.unet(x, t)
            actual = graph(x, t)
        assert graph.graph is not None
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
