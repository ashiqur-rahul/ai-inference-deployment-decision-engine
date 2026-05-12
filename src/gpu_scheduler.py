from __future__ import annotations
from dataclasses import dataclass

GPU_MEMORY_GB = {
    "CPU Server": 0,
    "NVIDIA T4": 16,
    "NVIDIA A10": 24,
    "NVIDIA A100 PCIe": 40,
    "NVIDIA H100 PCIe": 80,
    "Jetson AGX Xavier": 16,
}

@dataclass
class MemoryDecision:
    hardware: str
    estimated_memory_gb: float
    available_memory_gb: float
    fits: bool
    recommendation: str

def estimate_inference_memory_gb(model_size_mb: float, batch_size: int, activation_multiplier: float = 2.5) -> float:
    model_gb = float(model_size_mb) / 1024
    activation_gb = model_gb * activation_multiplier
    batch_overhead_gb = model_gb * max(int(batch_size) - 1, 0) * 0.35
    runtime_overhead_gb = 0.75
    return model_gb + activation_gb + batch_overhead_gb + runtime_overhead_gb

def evaluate_memory_fit(hardware: str, model_size_mb: float, batch_size: int) -> MemoryDecision:
    available = float(GPU_MEMORY_GB.get(hardware, 0))
    required = estimate_inference_memory_gb(model_size_mb, batch_size)
    fits = available == 0 or required <= available
    if available == 0:
        rec = "CPU path: GPU memory constraint not applicable."
    elif fits:
        rec = "Fits memory budget."
    else:
        rec = "Does not fit memory budget. Reduce batch size or select larger GPU."
    return MemoryDecision(hardware, required, available, bool(fits), rec)
