from PIL import Image

from convolve import Convolve

class LucasKanade:
    def __init__(self, im):
        self.im = im
        self.x_conv, self.y_conv = self.gradient()
        pass

    def gradient(self) -> tuple[Image.Image, Image.Image]:
        H_x_x = [[-1, 0, 1]]
        H_x_y = [[3], [10], [3]]
        H_y_x = [[3, 10, 3]]
        H_y_y = [[-1], [0], [1]]

        x_conv = Convolve.convolve(self.im, H_x_x)
        x_conv = Convolve.convolve(x_conv, H_x_y)

        y_conv = Convolve.convolve(self.im, H_y_x)
        y_conv = Convolve.convolve(y_conv, H_y_y)
        
        return (x_conv, y_conv)

    def jacobian(self) -> list[list[float]]:
        pass