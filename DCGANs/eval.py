import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

from IPython.display import HTML


with open("./pickles/images_L.pickle", "rb") as f:
    images_L = pickle.load(f)
with open("./pickles/losses_D.pickle", "rb") as f:
    losses_D = pickle.load(f)
with open("./pickles/losses_G.pickle", "rb") as f:
    losses_G = pickle.load(f)
