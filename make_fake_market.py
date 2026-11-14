"""Create a tiny random dataset in Market-1501 layout, only to check that the scripts run."""
import os
import sys

import numpy as np
from PIL import Image

root = sys.argv[1] if len(sys.argv) > 1 else "fake_market"
rng = np.random.default_rng(0)


def save(folder, pid, cam, n):
    os.makedirs(os.path.join(root, folder), exist_ok=True)
    base = rng.integers(0, 255, size=(1, 1, 3))
    for i in range(n):
        img = np.clip(base + rng.normal(0, 40, size=(128, 64, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(root, folder, f"{pid:04d}_c{cam}s1_{i:06d}_00.jpg"))


for pid in range(1, 21):
    for cam in (1, 2):
        save("bounding_box_train", pid, cam, 3)
for pid in range(101, 111):
    save("query", pid, 1, 1)
    save("bounding_box_test", pid, 2, 3)
    save("bounding_box_test", pid, 1, 1)
print("wrote", root)
