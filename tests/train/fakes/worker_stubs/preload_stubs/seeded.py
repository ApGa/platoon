"""Makes a numpy generator seeded from the OS: not fork-safe (every worker forked from the server would draw the same
numbers, where exec'd workers each draw their own)."""

import numpy as np

RNG = np.random.default_rng()
