class ScalarAccessor:
    def __init__(self, data):
        self.data = data

    def get_val(self, u, v):
        # Ensure indices are within the array bounds
        if 0 <= u < self.data.shape[1] and 0 <= v < self.data.shape[0]:
            return self.data[v, u]  # Note the order of indices (v, u)
        else:
            return 0  # Return some default value or handle boundary cases
            