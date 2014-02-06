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

    Formally, a binning along a single dimension is defined as follows:
    A bin with index i is the subset of the real line B_i = {x in R
    where l + i*t <= x < l + (i+1)*t}. The constants (l, u) are the
    lower and upper bounds, respectively, for that dimension, while t is
    the bin width, defined as t = (u - l) / c, where c is the number of
    bins along that dimension. The bins with indices 0 and (c-1) are
    specially defined as B_0 = {x in R where x < l + t} and
    B_(c-1) = {x in R where x >= u - t}, so that the binning paves the
    entire real line.

    In multiple dimensions, the bins are defined as intersections of the
    binnings for each individual dimension: A point (x_1, x_2,...,x_N)
    is considered within a bin (i_1,...,i_N) if each of the x_k are
    individually within the bins defined by the respective indices i_k.
    Bins are actually sequentially numbered rather than referred to by
    coordinate; this numbering is currently done via C-order enumeration
    of the multidimensional bin array (but is subject to change).

    Public methods:
        get_bin_num     Return the bin index for a given state

    """

    def __init__(self, low_bound, up_bound, bin_counts):
        """
        Define a new uniformly-spaced paving of phase space.

        Parameters:
            low_bound   Array of lower bounds, one for each coordinate
            up_bound    Array of upper bounds, one for each coordinate
            bin_counts  Number of bins in each coordinate direction

        The phase-space coordinates used should obviously be consistent
        (both in definition and in order) across the parameter arrays.

        """
        self.low_bound = np.asarray(low_bound)
        self.up_bound = np.asarray(up_bound)
        if any((self.up_bound - self.low_bound) <= 0):
            raise ValueError("Each upper bound must be greater than the " +
                             "corresponding lower bound.")
        self.bin_counts = np.asarray(bin_counts)
        if any(self.bin_counts <= 0):
            raise ValueError("Must specify at least one bin in each dimension")

    def get_bin_num(self, coords):
        """
        Return the bin number for a given point(s) in phase space.

        The bin number is a zero-based index of the available bins, but
        the only thing that really matters is that it is unique and
        consistent across multiple invocations on a single object.

        Parameters:
            coords  The coordinates of the point in phase space, using
                    the same coordinate system as in the constructor.

        Notes:

        Multiple sets of coordinates may be specified by passing a 2-D
        array in for coords. This array must be of the right shape so
        that the 1-D arrays such as low_bound and bin_counts specified
        during initialization may be correctly be broadcasted to it
        following NumPy's broadcasting rules.

        The return value in the case of multiple coordinates is an array
        of bin numbers, one for each point in phase space.

        """
        coords_norm = ((np.asarray(coords) - self.low_bound) /
                       (self.up_bound - self.low_bound))
        int_coords = np.empty(coords_norm.shape, dtype='int64')
        np.floor(coords_norm * self.bin_counts, int_coords)

        return np.ravel_multi_index(int_coords.transpose(), self.bin_counts,
                                    mode='clip')

