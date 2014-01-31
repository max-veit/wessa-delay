"""
Weighted-Ensemble method implementation.

Creates an ensemble of trajectories, using binning in the phase (or
state, or configuration) space to ensure more even sampling across
the entire space. This is particularly useful for getting information
about less-visited parts of the space.

Classes:
    Ensemble    The base encapsulation of a weighted ensemble
    Binning     Functionality related to binning of the phase space

"""

import numpy as np


