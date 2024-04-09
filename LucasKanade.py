from PIL import Image
import numpy as np
import math

from convolve import Convolve

P_SIZE = 6 # doing affine transformation

class LucasKanadeInverse:
    def __init__(self, I, R, eps, i_max):
        self.I = I
        self.R = R

        # Currently only deal with luminance images.
        self.I_arr = np.array(I, dtype=np.uint8)
        self.R_arr = np.array(R, dtype=np.uint8)

        self.eps = eps
        self.i_max = i_max
        self.n = P_SIZE
        self.p_init = np.array([1, 0, 0, 1, 0, 0]) # Shape according to textbook.
        self.p_opt = None 

        self.I_width, self.I_length = I.size
        self.R_width, self.R_length = R.size
        # Align with np array shape.
        # Reference point rather than used for accessing, so no need to round.
        self.R_center = np.array([0.5*self.R_length, 0.5*self.R_width])
        self.I_center = np.array([0.5*self.I_length, 0.5*self.I_width])

    def run(self) -> bool:
        #calculate gradient A.1
        R_x_grad_img, R_y_grad_img = self.gradient()
        R_x_grad = np.array(R_x_grad_img, dtype=np.float64)
        R_y_grad = np.array(R_y_grad_img, dtype=np.float64)
        print("Gradient calculation successful.")

        S = np.zeros((self.R_length, self.R_width, self.n))
        Hessian = np.zeros((self.n, self.n))

        for u in np.ndindex(self.R_arr.shape):
            coord = np.array(u) - self.R_center

            R_grad = np.array([R_x_grad[u], R_y_grad[u]]) # Column vector
            R_grad_row_vector = R_grad[np.newaxis, :] # Transpose to row vector

            j = self.jacobian_for_identity_affine(coord)
            s = R_grad_row_vector @ j
            S[u] = s.flatten()
            h = np.outer(s, s)
            Hessian += h

        try:
            Hessian_inv = np.linalg.inv(Hessian)
        except np.linalg.LinAlgError:
            return False # Hessian inversion failed
        
        print("Hessian inversion successful.")
        p = self.p_init.copy()
        i = 0

        while True:
            i += 1
            delta_p = np.zeros(self.n)
            print(f"Iteration {i}")
            
            for u in np.ndindex(self.R_arr.shape):

                R_coord = np.array(u) - self.R_center
                R_coord_prime = self.wrap(R_coord, p)

                I_coord = np.array(R_coord_prime) + self.I_center
                d = self.interpolate(self.I_arr, I_coord) - self.R_arr[u]
                s = S[u]
                delta_p += d * s

            q = Hessian_inv @ delta_p
            p_prime = self.optimize(p, q)
            if p_prime == None:
                return False
            
            p = p_prime.copy()
            loss = np.linalg.norm(q)
            print(f"Loss: {loss}")
            if loss <= self.eps or i >= self.i_max:
                break

        if i < self.i_max:
            self.p_opt = p.copy()
            return True
        else:
            return False

    def cartesian_to_homogeneous(self, coord: tuple) -> np.ndarray:
        y, x = coord
        arr = np.array([x, y, 1])

        return arr[:, np.newaxis] # Column vector

    def homogeneous_to_cartesian(self, coord: np.ndarray) -> tuple:
        x = coord[0][0]
        y = coord[1][0]
        z = coord[2][0]
        
        # np indexing
        if z == 0:
            return (y, x)
        else:
            return (y/z, x/z)
        
    def parameters_to_matrix(self, p: np.ndarray) -> np.ndarray:
        return np.array([[p[0], p[1], p[4]],
                         [p[2], p[3], p[5]],
                         [0, 0, 1]])
    
    def matrix_to_parameters(self, matrix: np.ndarray) -> np.ndarray:
        return np.array([matrix[0][0], matrix[0][1], 
                         matrix[1][0], matrix[1][1],  
                         matrix[0][2], matrix[1][2]])

    def gradient(self) -> tuple[Image.Image, Image.Image]:
        # S for Sobel
        S_x_x = [[-1, 0, 1]]
        S_x_y = [[1], [2], [1]]
        S_y_x = [[1, 2, 1]]
        S_y_y = [[-1], [0], [1]]

        x_conv = Convolve.convolve(self.R, S_x_x)
        x_conv = Convolve.convolve(x_conv, S_x_y)

        y_conv = Convolve.convolve(self.R, S_y_x)
        y_conv = Convolve.convolve(y_conv, S_y_y)
        
        return (x_conv, y_conv)

    def jacobian_for_identity_affine(self, coord: np.ndarray) -> np.ndarray:
        y, x = tuple(coord) # numpy indexing
        matrix = [[x, y, 0, 0, 1, 0], 
                  [0, 0, x, y, 0, 1]]
        
        return np.array(matrix)
    
    def wrap(self, coord: tuple, p: np.ndarray) -> tuple:
        xyz = self.cartesian_to_homogeneous(coord)
        matrix = self.parameters_to_matrix(p)
        
        xyz_prime = matrix @ xyz
        return self.homogeneous_to_cartesian(xyz_prime)
    
    def interpolate(self, img: np.ndarray, coord: np.ndarray) -> float:
        # Bilinear interpolation
        y, x = np.clip(coord, [0, 0], np.array(img.shape) - 1)
        xf, yf = int(x), int(y)
        xc, yc = math.ceil(x), math.ceil(y)
        
        # Boundaries handling
        yc = min(yc, img.shape[0] - 1)
        xc = min(xc, img.shape[1] - 1)
        
        # Interpolation weights
        dx, dy = x - xf, y - yf
        return (img[yf][xf] * (1 - dx) * (1 - dy) +
                img[yf][xc] * dx * (1 - dy) +
                img[yc][xf] * (1 - dx) * dy +
                img[yc][xc] * dx * dy)
    
    def optimize(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        A_p = self.parameters_to_matrix(p)
        A_q = self.parameters_to_matrix(q)
        try:
            A_q_inv = np.linalg.inv(A_q)
        except np.linalg.LinAlgError:
            return None

        A_p_prime = A_p @ A_q_inv
        
        return self.matrix_to_parameters(A_p_prime)