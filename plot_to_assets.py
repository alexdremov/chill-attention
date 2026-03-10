import os
import torch
import matplotlib.pyplot as plt
from chill_attention.mask import (
    FullChillMask,
    CausalChillMask,
    SlidingWindowChillMask,
    ChunkwiseChillMask,
    PrefixLMChillMask,
)

def plot_all_to_assets():
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    masks = [
        FullChillMask(),
        CausalChillMask(),
        SlidingWindowChillMask(16, 16),
        ChunkwiseChillMask(32, 1),
        PrefixLMChillMask(64),
    ]
    
    for mask in masks:
        print(f"Plotting {mask}...")
        # Plot with full range highlights
        fig = mask.plot(max_pos=256, plot_block_types=True, TILE_Q=32, TILE_K=32)
        save_path = os.path.join(assets_dir, f"mask_{mask.name.lower()}.png")
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_all_to_assets()
