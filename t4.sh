uv sync
uv run -m diana.data.download --categories hazelnut
uv run -m diana.profiler sweep --config configs/hazelnut.toml --device cuda --steps 50
uv run -m diana.train --config configs/hazelnut.toml --device cuda --use_amp --cudnn_benchmark --epochs 100
