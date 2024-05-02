from PIL import Image
import numpy as np
from convolve import Convolve
from matplotlib import pyplot as plt

P_SIZE = 6 # doing affine transformation

class LucasKanadeInverse:
    """
    Implements the Lucas-Kanade Inverse Compositional algorithm for image alignment using
    affine transformations. This class aligns a template image (R) to an input image (I)
    by iteratively adjusting the affine parameters to minimize the difference between
    the transformed template and the input image.

    Attributes:
        I (Image): The input image in which the template will be aligned.
        R (Image): The template image that needs to be aligned to the input image.
        eps (float): The convergence threshold for the optimization process.
        i_max (int): Maximum number of iterations to perform.
        p_init (np.ndarray): Initial parameters for the affine transformation.
        p_opt (np.ndarray): Optimized parameters after running the algorithm.
        losses (list): A list to store the loss at each iteration for monitoring convergence.
        iter_boundaries (list): List of boundaries at each iteration for visualization.
    """
    def __init__(self, I, R, eps, i_max, p_init):
        """
        Initializes the LucasKanadeInverse object with the input and reference images,
        convergence criteria, maximum number of iterations, and initial transformation parameters.

        Parameters:
            I (Image): The input image.
            R (Image): The reference or template image.
            eps (float): Convergence threshold.
            i_max (int): Maximum number of iterations allowed.
            p_init (np.ndarray): Initial guess of the affine parameters.
        """
        self.I = I
        self.R = R

        # Currently only deal with luminance images.
        self.I_arr = np.array(I, dtype=np.uint8)
        self.R_arr = np.array(R, dtype=np.uint8)

        self.eps = eps
        self.i_max = i_max
        self.total_iter = i_max
        self.n = P_SIZE
        self.p_init = p_init # Shape according to textbook.
        self.p_opt = None 
        self.losses = [] 

        self.I_width, self.I_length = I.size
        self.R_width, self.R_length = R.size
        # Align with np array shape.
        # Reference point rather than used for accessing, so no need to round.
        self.R_center = np.array([0.5*(self.R_length-1), 0.5*(self.R_width-1)])
        self.I_center = np.array([0.5*(self.I_length-1), 0.5*(self.I_width-1)])

        self.iter_boundaries = []

    def run(self) -> bool:
        """
        Executes the Lucas-Kanade algorithm to find the optimal affine parameters
        that align the template image to the input image. 

        Returns:
            bool: True if the algorithm converges within the specified number of iterations,
                  False otherwise.
        """
        #calculate gradient
        R_x_grad_img, R_y_grad_img = self.gradient()
        R_x_grad = np.array(R_x_grad_img, dtype=np.float64)
        R_y_grad = np.array(R_y_grad_img, dtype=np.float64)
        print("Gradient calculation successful.")

        #steepest descent image
        S = np.zeros((self.R_length, self.R_width, self.n))
        Hessian = np.zeros((self.n, self.n))

        for u in np.ndindex(self.R_arr.shape):
            coord = np.array(u) - self.R_center

            R_grad = np.array([R_x_grad[u], R_y_grad[u]]) # Column vector
            R_grad_row_vector = R_grad[np.newaxis, :] # Transpose to row vector

            j = self.jacobian_for_identity_affine(coord)
            s = R_grad_row_vector @ j
            S[u] = s
            h = np.outer(s, s)
            Hessian += h

        try:
            Hessian_inv = np.linalg.inv(Hessian)
        except np.linalg.LinAlgError:
            return False # Hessian inversion failed
        
        print("Hessian inversion successful.")
        p = self.p_init.copy()
        i = 0
        self.losses.clear()
        while True:
            i += 1
            delta_p = np.zeros(self.n)
            print(f"Iteration {i}")
            self.iter_boundaries.append([])
            
            for u in np.ndindex(self.R_arr.shape):

                R_coord = np.array(u) - self.R_center
                R_coord_prime = self.wrap(R_coord, p)
                I_coord = np.array(R_coord_prime) + self.I_center

                
                if self.check_boundary(u, self.R_arr.shape):
                    I_coord_round = self.round_and_check(I_coord, self.I_arr.shape)
                    self.iter_boundaries[i-1].append(I_coord_round)
                
                d = self.interpolate(self.I_arr,I_coord) - self.R_arr[u]
                s = S[u]
                delta_p += d * s

            q = Hessian_inv @ delta_p

            p_prime = self.optimize(p, q)

            if p_prime is None:
                return False

            p = p_prime.copy()
            loss = np.linalg.norm(q)
            self.losses.append(loss)
            print(f"Loss: {loss}")
            if loss <= self.eps or i >= self.i_max:
                break

        if i < self.i_max:
            self.p_opt = p.copy()
            self.total_iter = i

            return True
        else:
            return False
        
    def check_boundary(self, coord: tuple, shape: tuple) -> bool:
        """
        Checks if the given coordinate is on the boundary of the image.

        Parameters:
            coord (tuple): The coordinate (row, column) to check.
            shape (tuple): The dimensions (rows, columns) of the image.

        Returns:
            bool: True if the coordinate is on the boundary, False otherwise.
        """
        rows, cols = shape
        i, j = coord

        return i == 0 or j == 0 or i == rows-1 or j == cols-1
    
    def round_and_check(self, coord: np.ndarray, reference_shape: tuple):
        """
        Rounds coordinates to integer values and checks if they are within the image boundaries.

        Parameters:
            coord (np.ndarray): The coordinate to round and check.
            reference_shape (tuple): The dimensions (rows, columns) of the reference image.

        Returns:
            tuple: Rounded and checked coordinate.
        """
        # Step 1: Round the coordinates
        rounded_coord = np.round(coord).astype(int)
        
        # Step 2: Convert to tuple
        rounded_coord_tuple = tuple(rounded_coord)
        
        # Step 3: Ensure the coordinates are within the array's range
        # Clipping each coordinate to the range [0, dimension_size - 1]
        checked_coord = (min(max(rounded_coord_tuple[0], 0), reference_shape[0] - 1),
                        min(max(rounded_coord_tuple[1], 0), reference_shape[1] - 1))
        
        return checked_coord

    def cartesian_to_homogeneous(self, coord: tuple) -> np.ndarray:
        """
        Converts Cartesian coordinates to homogeneous coordinates.

        Parameters:
            coord (tuple): A tuple of (y, x) coordinates.

        Returns:
            np.ndarray: Corresponding homogeneous coordinates as a column vector.
        """
        y, x = coord
        arr = np.array([x, y, 1])

        return arr[:, np.newaxis] # Column vector

    def homogeneous_to_cartesian(self, coord: np.ndarray) -> tuple:
        """
        Converts homogeneous coordinates back to Cartesian coordinates.

        Parameters:
            coord (np.ndarray): Homogeneous coordinates as a column vector.

        Returns:
            tuple: Corresponding Cartesian coordinates (y, x).
        """
        x = coord[0][0]
        y = coord[1][0]
        z = coord[2][0]
        
        # np indexing
        if z == 0:
            return (y, x)
        else:
            return (y/z, x/z)
        
    def parameters_to_matrix(self, p: np.ndarray) -> np.ndarray:
        """
        Converts a parameter vector into an affine transformation matrix.

        Parameters:
            p (np.ndarray): Parameter vector representing the affine transformation.

        Returns:
            np.ndarray: Corresponding 3x3 affine transformation matrix.
        """

        return np.array([[p[0]+1, p[1], p[4]],
                         [p[2], p[3]+1, p[5]],
                         [0, 0, 1]])
    
    def matrix_to_parameters(self, matrix: np.ndarray) -> np.ndarray:
        """
        Converts an affine transformation matrix back to a parameter vector.

        Parameters:
            matrix (np.ndarray): A 3x3 affine transformation matrix.

        Returns:
            np.ndarray: Corresponding parameter vector.
        """
        return np.array([matrix[0][0]-1, matrix[0][1], 
                         matrix[1][0], matrix[1][1]-1,  
                         matrix[0][2], matrix[1][2]])

    def gradient(self) -> tuple[Image.Image, Image.Image]:
        """
        Computes the gradient of the template image using a Sobel operator.

        Returns:
            tuple[Image.Image, Image.Image]: Gradient images in x and y directions.
        """
        # S for Sobel
        S_x_x = [[-1, 0, 1]]
        S_x_y = [[3], [10], [3]]
        S_y_x = [[3, 10, 3]]
        S_y_y = [[-1], [0], [1]]

        x_conv = Convolve.convolve(self.R, S_x_x)
        x_conv = Convolve.convolve(x_conv, S_x_y)

        y_conv = Convolve.convolve(self.R, S_y_x)
        y_conv = Convolve.convolve(y_conv, S_y_y)
        
        return (x_conv, y_conv)

    def jacobian_for_identity_affine(self, coord: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian matrix of the affine transformation at a given coordinate.

        Parameters:
            coord (np.ndarray): The coordinate (y, x) at which to evaluate the Jacobian.

        Returns:
            np.ndarray: Jacobian matrix for the identity-affine transformation.
        """
        y, x = tuple(coord) # numpy indexing
        matrix = [[x, y, 0, 0, 1, 0], 
                  [0, 0, x, y, 0, 1]]
        
        return np.array(matrix)
    
    def wrap(self, coord: tuple, p: np.ndarray) -> tuple:
        """
        Applies an affine transformation to a given coordinate using the current parameters.

        Parameters:
            coord (tuple): Original Cartesian coordinates (y, x).
            p (np.ndarray): Current affine transformation parameters.

        Returns:
            tuple: Transformed coordinates (y, x) after applying the affine transformation.
        """
        xyz = self.cartesian_to_homogeneous(coord)
        matrix = self.parameters_to_matrix(p)
        
        xyz_prime = matrix @ xyz
        return self.homogeneous_to_cartesian(xyz_prime)
    
    def interpolate(self, img: np.ndarray, coord: np.ndarray) -> float:
        """
        Performs bilinear interpolation to estimate the intensity at non-integer coordinates.

        Parameters:
            img (np.ndarray): Image array in which to interpolate.
            coord (np.ndarray): The coordinate (y, x) at which to interpolate.

        Returns:
            float: Interpolated intensity value at the given coordinate.
        """
        # Ensure the coordinate is within image bounds
        y, x = np.clip(coord, [0, 0], [img.shape[1] - 1, img.shape[0] - 1])
        xf, yf = int(x), int(y)
        xc, yc = xf+1, yf+1
        
        # Ensure coordinates do not exceed image dimensions
        xc = min(xc, img.shape[1] - 1)
        yc = min(yc, img.shape[0] - 1)
        a = x-xf
        b = y-yf

        A = float(img[yf, xf])
        B = float(img[yf, xc])
        C = float(img[yc, xf])
        D = float(img[yc, xc])
        E = A + a * (B - A)
        F = C + a * (D - C)
        G = E + b * (F - E)

        return float(G)
    
    def optimize(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Updates the parameter vector using an inverse compositional step.

        Parameters:
            p (np.ndarray): Current parameter vector.
            q (np.ndarray): Increment vector computed from the gradient.

        Returns:
            np.ndarray: Updated parameter vector after applying the increment.
        """
        A_p = self.parameters_to_matrix(p)
        A_q = self.parameters_to_matrix(q)
        try:
            A_q_inv = np.linalg.inv(A_q)
        except np.linalg.LinAlgError:
            return None

        A_p_prime = A_p @ A_q_inv
        
        return self.matrix_to_parameters(A_p_prime)
    

    def plot_loss_curve(self):
        """
        Plots the loss curve of the Lucas-Kanade optimization process.
        Saves the plot as 'loss_curve.png'.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label='Loss over iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Curve of Lucas-Kanade Inverse')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_curve.png')

    def boundary_visualize(self, iter: int, color: tuple) -> Image.Image:
        """
        Visualizes the boundaries at a given iteration of the optimization process on the input image.

        Parameters:
            iter (int): The iteration number to visualize.
            color (tuple): RGB tuple for the boundary color.

        Returns:
            Image.Image: The input image with the boundaries highlighted.
        """
        boundary_list = self.iter_boundaries[iter]
        boundary_img = self.I.convert("RGB")

        for (v, u) in boundary_list:
            boundary_img.putpixel((u, v), color)
        
        return boundary_img

