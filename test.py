import wisardpkg as wp
import numpy as np

addressSize = 3    # number of addressing bits in the ram.
minScore = 0.1  # min score of training process
threshold = 10   # limit of training cycles by discriminator
discriminatorLimit = 5    # limit of discriminators by clusters

# False by default for performance reasons
# when enabled,e ClusWiSARD prints the progress of train() and classify()
verbose = True

clus = wp.ClusWisard(addressSize, minScore, threshold,
                     discriminatorLimit, verbose=True)

X = [
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1]
]

y = [
    "cold",
    "cold",
    "hot",
    "hot"
]

y2 = {
    1: "cold",
    2: "hot"
}

breakpoint()
# train using the input data
clus.train(X, y2)

# optionally you can train using arbitrary labels for the data
# input some labels in a dict,
# the keys must be integer indices indicating which input array the entry is associated to,
# the values are the labels which must be strings

# classify some data
out = clus.classify(X)

# the output of classify is a string list in the same sequence as the input
for i, d in enumerate(X):
    print(out[i], d)
