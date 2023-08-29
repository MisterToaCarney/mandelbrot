import numpy as np
import numexpr as ne
import cv2

class Interactive:
    def __init__(self):
        self.center = [-0.7492, 0.07565]
        self.span = 1
        self.resolution = (1500,1500)
        self.iterations = 4000
        self.scale_factor = 0.66666

        self.running = True

        self.idxs = self.init_idxs()
        self.initialize()
        
    def get_extent(self):
        return [
            self.center[0] - self.span, 
            self.center[0] + self.span, 
            self.center[1] - self.span, 
            self.center[1] + self.span
        ]
    
    def initialize(self):
        self.iteration = 0
        self.c = self.init_c()
        self.z = self.init_z()
        self.image = self.init_image()
    
    def flat_shape(self):
        return (self.resolution[0] * self.resolution[1],)

    def init_c(self):
        extent = self.get_extent()
        real = np.linspace(extent[0], extent[1], self.resolution[1], dtype=np.float64)
        imag = (np.linspace(extent[3], extent[2], self.resolution[0], dtype=np.float64) * 1j)[None,:].T
        c = real + imag
        c_flat = c.flatten()
        return c_flat
    
    def init_idxs(self):
        idxs = np.fromiter(np.ndindex(self.resolution), dtype=np.dtype((int, 2)))
        return idxs
    
    def init_z(self):
        z = np.full(self.flat_shape(), 0+0j)
        return z

    def init_image(self):
        image = np.zeros(self.resolution, dtype=np.int64)
        return image
    
    def display_image(self):
        image_normalized = self.image.copy()
        image_normalized[image_normalized == 0] = self.iteration
        image_normalized = np.log(image_normalized)
        image_normalized = cv2.normalize(image_normalized, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        image_normalized = cv2.applyColorMap(image_normalized, cv2.COLORMAP_INFERNO)
        image_normalized = cv2.resize(image_normalized, (0,0), fx=self.scale_factor, fy=self.scale_factor)
        cv2.imshow('brot', image_normalized)
        
    
    def pan(self, direction):
        if direction == 'up':
            self.center[1] += 0.4*self.span
        elif direction == 'down':
            self.center[1] -= 0.4*self.span
        elif direction == 'left':
            self.center[0] -= 0.4*self.span
        elif direction == 'right':
            self.center[0] += 0.4*self.span
        else:
            print('Unknown direction!')
        self.initialize()
        print(self.center)
    
    def zoom(self, direction):
        if direction == 'in':
            self.span = self.span / 2
        elif direction == 'out':
            self.span = self.span * 2
        self.initialize()
        print(1/self.span)
            

    def handle_key(self, key):
        if key == 65361: self.pan('left')
        elif key == 65362: self.pan('up')
        elif key == 65363: self.pan('right')
        elif key == 65364: self.pan('down')
        elif key == 65365: self.zoom('in')
        elif key == 65366: self.zoom('out')
        elif key == 113: self.running = False
        elif key == -1: pass
        else: print(key)

    def run(self):
        self.iteration = 0
        while self.running:
            while self.iteration < self.iterations and self.running:
                self.z = ne.evaluate("z**2 + c", local_dict={'z': self.z, 'c': self.c})
                mask = ne.evaluate("abs(z).real > 2", local_dict={'z': self.z})

                target_indexes = self.idxs[mask]
                self.image[target_indexes[:,0], target_indexes[:,1]] = self.iteration

                if self.iteration % 5 == 0 and (np.count_nonzero(self.image) > 0): 
                    self.display_image()
                    self.handle_key(cv2.waitKeyEx(1))

                self.iteration += 1
            
            self.handle_key(cv2.waitKeyEx(100))

        cv2.destroyAllWindows()


if __name__ == "__main__":
    runner = Interactive()
    runner.run()

