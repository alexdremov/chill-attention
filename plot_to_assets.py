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
        # Use smaller max_pos and TILE_Q to make highlights more visible
        max_pos = 128
        TILE_Q = 16
        TILE_K = 16
        
        if mask.has_k_full_range():
            # Debug: print full ranges
            ranges = mask._calc_full_ranges(max_pos, TILE_Q)
            print(f"Full ranges for {mask.name}:\n{ranges}")

        fig = mask.plot(max_pos=max_pos, plot_block_types=True, TILE_Q=TILE_Q, TILE_K=TILE_K)
        save_path = os.path.join(assets_dir, f"mask_{mask.name.lower()}.png")
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_all_to_assets()
