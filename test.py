import numpy as np
import wisardpkg as wp
import pandas as pd

addressSize = 3     # number of addressing bits in the ram
ignoreZero = False  # optional; causes the rams to ignore the address 0

# False by default for performance reasons,
# when True, WiSARD prints the progress of train() and classify()
verbose = False

wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)


X = [
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1]
]

# load label data, which must be a string array
y = [
    "cold",
    "cold",
    "hot",
    "hot"
]

wsd.train(X, y)

X = [
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1]
]

# the output is a list of string, this represent the classes attributed to each input
out = wsd.classify(X)
breakpoint()
for oneout in out:
    print(oneout)
