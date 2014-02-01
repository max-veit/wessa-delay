"""
Weighted-Ensemble method implementation.

Creates an ensemble of trajectories, using binning in the phase (or
state, or configuration) space to ensure more even sampling across
the entire space. This is particularly useful for getting information
about less-visited parts of the space.

Classes:
    Ensemble    The base encapsulation of a weighted ensemble
    Paving      Functionality related to binning of the phase space

"""

import numpy as np


class Ensemble(object):
    pass


# TODO Make abstract class?
class Paving(object):
    """
    Functionality associated with a paving of phase space.

    Public methods:
        get_bin_num     Return the bin index for a given state

    """

    def __init__(self):
        raise NotImplementedError("Abstract class.")

    def get_bin_num(self):
        raise NotImplementedError("Abstract class.")


class UniformPaving(Paving):
    """
    A paving of phase space that is uniform in every dimension.

    The binning is delimited by a rectangular region in phase space.
    Coordinates outside this region are considered to belong to the
    bin closest to that point.

    Public methods:
        get_bin_num     Return the bin index for a given state

    """

    def __init__(self, low_bound, up_bound, bin_counts):
        pass

    def get_bin_num(self, coords):
        pass
