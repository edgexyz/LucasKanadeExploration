from LucasKanade import LucasKanadeInverse
from PIL import Image

I = Image.open("man_distort.png")
if I.mode != "L":
    I = I.convert("L")
R = Image.open("man.png")
if R.mode != "L":
    R = R.convert("L")
eps = 0.001
i_max = 1000
LK = LucasKanadeInverse(I, R, eps, i_max)
LK.run()