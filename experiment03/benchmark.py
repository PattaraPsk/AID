# from ai_benchmark import AIBenchmark
# benchmark = AIBenchmark()
# results = benchmark.run()

import torch
from torchvision.models import efficientnet_b0
from pytorch_benchmark import benchmark


model = efficientnet_b0()
sample = torch.randn(8, 3, 224, 224)  # (B, C, H, W)
results = benchmark(model, sample, num_runs=100)