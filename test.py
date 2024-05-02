from LucasKanade import LucasKanadeInverse
from PIL import Image

# Test code for Lucas-Kanade Inverse Algorithm

# Import Search Image Here
I = Image.open("man.png")
if I.mode != "L":
    I = I.convert("L")
# Import Reference Image Here
R = Image.open("man_distort.png")
if R.mode != "L":
    R = R.convert("L")

# Run Lucas-Kanade Inverse Algorithm
eps = 0.001 # epsilon
i_max = 1000 # maximum number of iterations
LK = LucasKanadeInverse(I, R, eps, i_max)
if LK.run():
    LK.plot_loss_curve()
else:
    print("Failed to converge")


