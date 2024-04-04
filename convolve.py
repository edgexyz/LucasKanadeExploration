from PIL import Image

class Convolve:
    def check_range(loc, range):
        """
        Ensures that a given location value stays within a specified range by mirroring
        the value at the boundaries if it falls outside.

        Parameters:
        - loc (int): The location value to check.
        - range (int): The maximum range (exclusive) the location should fall within.

        Returns:
        - int: A location value adjusted to stay within the specified range.
        """
        while loc < 0 or loc >= range:
            if loc < 0:
                loc = -loc
            if loc >= range:
                loc = 2*(range-1) - loc
        
        return loc

    def convolve(im, kernel):
        """
        Applies convolution to an image using a given kernel, normalizing the result
        based on the sum of the kernel's values.

        Parameters:
        - im (PIL.Image.Image): The image to convolve.
        - kernel (list of list of int): The convolution kernel.

        Returns:
        - PIL.Image.Image: The convolved image.
        """
        K = len(kernel[0]) // 2
        L = len(kernel) // 2

        kernel_sum = sum([abs(element) for row in kernel for element in row])
        norm = float(1 / kernel_sum)

        width, height = im.size
        convolved = Image.new("F", (width, height))

        for v in range(height):
            for u in range(width):
                pixel_sum = 0
                for l in range(-L, L+1):
                    y = Convolve.check_range(v+l, height)
                    for k in range(-K, K+1):
                        x = Convolve.check_range(u+k, width)
                        
                        pixel_sum += (im.getpixel((x, y)) * kernel[L+l][K+k])

                pixel = pixel_sum * norm
                convolved.putpixel((u, v), pixel)
        
        return convolved
    

# test = [[-1], [0], [1]]
# print(sum([abs(element) for row in test for element in row]))