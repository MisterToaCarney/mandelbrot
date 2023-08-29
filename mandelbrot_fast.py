import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import time

plt.rcParams['figure.figsize'] = [10,10]

center = (-0.7492226869, 0.0756571369)
span = 0.001
extent = [center[0] - span, center[0] + span, center[1] - span, center[1] + span]
resolution = (1000,1000)
iterations = 4000

real = np.linspace(extent[0], extent[1], resolution[1], dtype=np.float32)
imag = (np.linspace(extent[3], extent[2], resolution[0], dtype=np.float32) * 1j)[None,:].T
c = real + imag
c_flat = c.flatten()

idxs = np.fromiter(np.ndindex(resolution), dtype=np.dtype((int, 2)))
z = np.full(c_flat.shape, 0+0j)
image = np.zeros(resolution, dtype=np.int64)

start = time.monotonic()
for iteration in range(iterations):
    z = ne.evaluate("z**2 + c_flat")
    mask = ne.evaluate("abs(z).real > 2")
    
    target_indexes = idxs[mask]
    image[target_indexes[:,0], target_indexes[:,1]] = iteration
    
    if np.count_nonzero(mask) > 0:
        idxs = np.delete(idxs, mask, axis=0)
        c_flat = np.delete(c_flat, mask, axis=0)
        z = np.delete(z, mask, axis=0)

end = time.monotonic()
print("Completed", iterations, "iterations in", end-start, "seconds")

image[image == 0] = iterations
image = np.log(image)

plt.imshow(image, cmap='inferno', extent=extent)
plt.show()
