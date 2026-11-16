"""Contoh sintetis untuk MENGUJI model, bukan hasil pengukuran WiFi."""
from __future__ import annotations

from inference import SOURCE_COLUMNS

PERTURBATIONS = (-0.4, 0.2, 0.3, -0.6, 0.0, 0.4, 0.6, -0.2, 0.1, -0.1, 0.3, 0.2)


def create_scenario(name: str) -> list[dict[str, float]]:
    if name == 'normal':
        latency, jitter, packet_loss, throughput = 18.0, 2.0, 0.0, 25.0
    elif name == 'gangguan':
        latency, jitter, packet_loss, throughput = 200.0, 50.0, 12.0, 2.0
    else:
        raise ValueError('Skenario tidak dikenal.')
    result = []
    for fluctuation in PERTURBATIONS:
        result.append({
            'latency_ms': round(max(0, latency + fluctuation * 2), 3),
            'jitter_ms': round(max(0, jitter + fluctuation * 0.35), 3),
            'packet_loss_percent': round(max(0, min(100, packet_loss + fluctuation * 0.3)), 3),
            'throughput_mbps': round(max(0.05, throughput + fluctuation * 0.7), 3),
        })
    return result


def to_dataframe(samples):
    import pandas as pd
    return pd.DataFrame(samples, columns=list(SOURCE_COLUMNS))
