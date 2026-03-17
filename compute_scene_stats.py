import h5py
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

def compute_stats():
    index_path = Path("data/mumucd/.patch_index.json")
    if not index_path.exists():
        print("Index not found")
        return

    with open(index_path, "r") as f:
        data = json.load(f)

    data_root = Path("data/mumucd")
    scene_paths = [data_root / p for p in data["scene_paths"]]
    
    stats = []
    print(f"Computing stats for {len(scene_paths)} scenes...")
    for path in tqdm(scene_paths):
        try:
            with h5py.File(path, "r") as f:
                # sr shape is (C, H, W) or (H, W, C)
                ds = f["sr"]
                # We need per-band min/max
                # Using a small chunking or global min/max if possible
                # For safety, we load and compute. 
                # Since we have 64GB RAM, we can afford to load one 1GB scene at a time here.
                arr = ds[()] 
                # Ensure CHW for normalization
                # Based on _to_chw logic: channel is smallest
                channel_axis = int(np.argmin(arr.shape))
                if channel_axis == 0:
                    pass
                elif channel_axis == 1:
                    arr = np.transpose(arr, (1, 0, 2))
                else:
                    arr = np.transpose(arr, (2, 0, 1))
                
                mn = arr.min(axis=(-2, -1)).tolist()
                mx = arr.max(axis=(-2, -1)).tolist()
                stats.append({"min": mn, "max": mx})
        except Exception as e:
            print(f"Error processing {path}: {e}")
            # Fallback
            stats.append(None)

    data["scene_stats"] = stats
    with open(index_path, "w") as f:
        json.dump(data, f)
    print("Done! stats saved to .patch_index.json")

if __name__ == "__main__":
    compute_stats()
