class ScalarAccessor:
    """
    A class designed to safely access elements of a 2D numpy array. It provides boundary checks
    to prevent out-of-bounds errors when accessing the array.

    Attributes:
        data (np.ndarray): The 2D numpy array from which values are to be accessed.
    """
    def __init__(self, data):
        """
        Initializes the ScalarAccessor object with a 2D numpy array.

        Parameters:
            data (np.ndarray): The 2D numpy array to be accessed.
        """
        self.data = data

    def get_val(self, u, v):
        """
        Retrieves the value at specified indices from the 2D array, with boundary checks.

        Parameters:
            u (int): The horizontal index (column index) into the data array.
            v (int): The vertical index (row index) into the data array.

        Returns:
            The value at the given indices if within bounds, otherwise returns 0 as a default value.
        """
        # Ensure indices are within the array bounds
        if 0 <= u < self.data.shape[1] and 0 <= v < self.data.shape[0]:
            return self.data[v, u]  # Note the order of indices (v, u)
        else:
            return 0  # Return some default value or handle boundary cases
            