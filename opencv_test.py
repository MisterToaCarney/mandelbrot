import numpy as np
import numexpr as ne
import cv2

center = (-0.7492, 0.0756571369)
span = 0.0001
extent = [center[0] - span, center[0] + span, center[1] - span, center[1] + span]
resolution = (1000,1000)
iterations = 4000

real = np.linspace(extent[0], extent[1], resolution[1], dtype=np.float64)
imag = (np.linspace(extent[3], extent[2], resolution[0], dtype=np.float64) * 1j)[None,:].T
c = real + imag
c_flat = c.flatten()

idxs = np.fromiter(np.ndindex(resolution), dtype=np.dtype((int, 2)))
z = np.full(c_flat.shape, 0+0j)
image = np.zeros(resolution, dtype=np.int64)

iteration = 0

while True:
    while iteration < iterations:
        iteration += 1

        z = ne.evaluate("z**2 + c_flat")
        mask = ne.evaluate("abs(z).real > 2")
        
        target_indexes = idxs[mask]
        image[target_indexes[:,0], target_indexes[:,1]] = iteration

        image_normalized = image.copy()
        image_normalized[image_normalized == 0] = iteration
        image_normalized = np.log(image_normalized)
        image_normalized = cv2.normalize(image_normalized, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

        image_normalized = cv2.applyColorMap(image_normalized, cv2.COLORMAP_INFERNO)
        cv2.imshow('brot', image_normalized)

        key = cv2.waitKeyEx(1)
        if key == 65361: 
            iteration = 0

        elif key == 65362: pass
        elif key == 65363: pass
        elif key == 65364: pass
        elif key == 113: 
            iteration = iterations
        else: print(key)

        
        if np.count_nonzero(mask) > 0:
            idxs = np.delete(idxs, mask, axis=0)
            c_flat = np.delete(c_flat, mask, axis=0)
            z = np.delete(z, mask, axis=0)
    

print("Complete")

while True:
    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyAllWindows()