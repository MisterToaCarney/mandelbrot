import torch
import numpy as np
import time
import matplotlib.pyplot as plt

device = torch.device('cuda')

center = (-0.7492226869, 0.0756571369)
# center = (-1.499328, 0)
span = 0.001
extent = [center[0] - span, center[0] + span, center[1] - span, center[1] + span]
resolution = (1000,1000)
iterations = 4000

real = torch.linspace(extent[0], extent[1], resolution[1], device=device, dtype=torch.float32)
imag = (torch.linspace(extent[3], extent[2], resolution[0], device=device, dtype=torch.float32) * 1j)[None,:].T
c = real + imag
c_flat = c.flatten()

idxs = np.fromiter(np.ndindex(resolution), dtype=np.dtype((int, 2)))
idxs = torch.tensor(idxs, device=device)

z = torch.full(c_flat.shape, 0+0j, device=device)
image = torch.zeros(resolution, dtype=torch.int64, device=device)

full_mask = torch.ones(c_flat.shape, dtype=bool, device=device)


print("Starting")
start = time.monotonic()
for iteration in range(iterations):
    z[full_mask] = z[full_mask]**2 + c_flat[full_mask]
    mask = torch.abs(z[full_mask]) > 2

    if torch.count_nonzero(mask) == 0: continue

    target_indexes = idxs[full_mask][mask]
    image[target_indexes[:,0], target_indexes[:,1]] = iteration
    full_mask[full_mask == True] = full_mask[full_mask == True] & ~mask


end = time.monotonic()
print("Resolution:", resolution, "Iterations:", iterations, "calculated in", end-start, "seconds")

image[image == 0] = iterations
final_image = torch.log(image).cpu()

plt.imshow(final_image, cmap='inferno', extent=extent)
plt.show()

