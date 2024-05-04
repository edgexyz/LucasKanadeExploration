# Lucas Kanade Exploration
<img width="645" alt="截屏2024-05-04 下午2 48 25" src="https://github.com/edgexyz/LucasKanadeExploration/assets/90734286/756e7882-7693-44ff-93f5-fce0dec08fab">

by Jerry Li and Shu Yang

## Description
This is a Python implementation of the Inverse Compositional Algorithm based on the Lucas-Kanade method. The implementation is designed for image alignment tasks, particularly using affine transformations to align a template image to an input image. It offers a robust approach to iteratively adjust the affine parameters to minimize the difference between the transformed template and the input image, ensuring optimal alignment.

## Geometry of the Inverse Compositional Algorithm
<img width="549" alt="截屏2024-05-04 下午2 50 19" src="https://github.com/edgexyz/LucasKanadeExploration/assets/90734286/5e6bab6d-0677-4e52-bd9f-ae2be4f85c39">

With the reference image $R$ fixed, the Inverse Compositional Algorithm aims to find the geometric transformation that warps the search image $I$ to match the reference image. The algorithms starts with an initial transformation $p$ that warps $I$ into $I_p$ centered at origin. Then iteratively updating the transformation parameter $q$.

## Prerequisites
- Python Pillow library
- Numpy
- Matplotlib.plt

## How to Use
1. Clone this repository
2. Add the image you want to match in the folder
3. Go to [demo](demo.ipynb) and follow the instruction in the notebook.

