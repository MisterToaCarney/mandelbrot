import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import time

plt.rcParams['figure.figsize'] = [10,10]

center = (-0.7492226315, 0.07565701935)
span = 0.000001
extent = [center[0] - span, center[0] + span, center[1] - span, center[1] + span]
dims = (1000,1000)
iterations = 2000

real = np.linspace(extent[0], extent[1], dims[1])
imag = (np.linspace(extent[3], extent[2], dims[0]) * 1j)[None,:].T
c = real + imag

z = np.full((dims), 0+0j, dtype=np.complex128)
out = np.full((dims), 0, dtype=np.int64)

start = time.monotonic()

for iteration in range(iterations):
    # z = z**2 + c
    z = ne.evaluate("z**2 + c")

    # out += ((out == 0) & (np.absolute(z) > 2)).astype(np.uint8) * iteration
    # out[(out == 0) & (np.absolute(z) > 2)] = iteration
    out += ne.evaluate("((out == 0) & (abs(z).real > 2)) * iteration")
    print(iteration)

end = time.monotonic()

out[out==0] = iterations
out = np.log(out)

print("Computed", iterations, "iterations in", end-start, "seconds")

plt.xlabel('real')
plt.ylabel('imaginary')
plt.imshow(out, cmap='inferno', extent=extent)

plt.imsave('2.png', out, cmap='inferno')

plt.show()

