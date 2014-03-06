"""
Weighted-Ensemble method implementation.

Creates an ensemble of trajectories, using binning in the phase (or
state, or configuration) space to ensure more even sampling across
the entire space. This is particularly useful for getting information
about less-visited parts of the space.

Throughout this file, the term "phase space" is used as a proxy for the
space in which the system is evolving (i.e. the space it is exploring),
be it classical phase space, chemical concentration space, or anything
else along the same lines.

Classes:
    WeightedTrajectory
                A trajectory with a weight attached to it.
    Ensemble    The base encapsulation of a weighted ensemble
    Paving      Functionality related to binning of the phase space

"""


import functools
from heapq import heappush, heappop
from collections import defaultdict
from copy import copy, deepcopy

import numpy as np

from ssad import Trajectory


@functools.total_ordering
class WeightedTrajectory(Trajectory):

    """
    A trajectory with a weight and comparison operators based on weight.

    Inherits ssad.Trajectory. The only changes are summarized below:

    Public attributes:
        weight  The statistical weight of this trajectory.

    Public methods:
        clone   Override Trajectory.clone; make copies of this traj.

    Comparison operators are implemented and are based on weight.
    Two weighted trajectories compare exactly as their weights,
    e.g. (wtrj1 <= wtrj2) == (wtrj1.weight <= wtrj2.weight).

    """

    # TODO Make weight a NumPy precision float
    def __init__(self, state, reactions, weight, init_time=0.0):
        """
        Initialize a new weighted trajectory.

        Parameters:
            state       Initial state vector. Indices represent species
                        indices, values are the populations of those
                        species.
            reactions   List of reaction pathways that govern this
                        trajectory's dynamics.
            weight      Initial statistical weight of this trajectory.
                        Defaults to 1.

        Optional Parameters:
            init_time   Simulation time at which this trajectory starts.
                        Defaults to 0.0 time units.

        """
        super(WeightedTrajectory, self).__init__(
                state, reactions, init_time)
        self.weight = weight

    def clone(self, num_clones, weights=None):
        """
        Copy this trajectory to obtain num_clones _extra_ trajectories.

        This method behaves similarly to Trajectory.clone, except
        weights are handled as described below.

        Parameters:
            num_clones  The number of additional trajectories to
                        produce. The total number of identical
                        trajectories will then be num_clones + 1.

        Optional Parameters:
            weights     The weights to assign to the group of
                        trajectories. May be either a number, a list of
                        length (num_clones + 1), or None.
                        If None, each trajectory (including this one) is
                        assigned the weight
                        self.weight / (num_clones. + 1).
                        If a number, each trajectory is assigned that
                        number as a weight.
                        If a list, each trajectory will be assigned a
                        unique element of the list as a weight. This
                        trajectory will take the first element of the
                        list while the clones will be assigned the
                        remaining elements.
                        Default None.

        Returns:
            A list of trajectories of length num_clones. If weights was
            specified as a list, the weights of the output trajectories
            will be in the same order as that list.

        """
        if num_clones < 1:
            raise ValueError("Must specify a positive number of clones.")
        if (weights is not None and
                not np.isscalar(weights) and
                np.asarray(weights).size != num_clones + 1):
            raise ValueError("Weight list must have length equal to the " +
                             "number of clones plus one.")
        if weights is None:
            weights = self.weight / (num_clones + 1)
        clones = []
        for cidx in range(num_clones + 1):
            if np.isscalar(weights):
                new_weight = weights
            else:
                new_clone.weight = weights[cidx]
            if cidx == 0:
                self.weight = new_weight
                continue
            new_clone = WeightedTrajectory(self.state,
                                           self.reactions,
                                           new_weight,
                                           init_time=self.time)
            # A deep copy is probably not necessary here, as the past
            # history should not be modified.
            # TODO Consider selective omission of history
            new_clone.hist_times = copy(self.hist_times)
            new_clone.hist_states = copy(self.hist_states)
            new_clone.next_rxn = self.next_rxn
            new_clone.next_rxn_time = self.next_rxn_time
            new_clone.last_rxn_time = self.last_rxn_time
            clones.append(new_clone)
        return clones

    def __lt__(self, other):
        return (self.weight < other.weight)

    def __eq__(self, other):
        return (self.weight == other.weight)

    def __str__(self):
        msg = super(WeightedTrajectory, self).__str__()
        msg = ' '.join((msg, "Weight", str(self.weight) + "."))
        return msg

class Ensemble(object):

    """
    An ensemble of weighted trajectories in phase space.

    The weighted-ensemble method allows one to more completely and
    accurately sample the probability distribution of a system over
    phase space. The method uses periodic resampling to equalize the
    distribution of samples over phase space, obtaining a more accurate
    estimate of the distribution in less-visited regions of the space.

    Public methods:
        run_step        Run one step of the weighted-ensemble algorithm.
        run_time        Run weighted-ensemble for a specified sim. time.

    Also implements the iterator protocol, which iterates over all the
    trajectories in the ensemble (in no predetermined order).

    Public attributes:
        coords          List of coords of all traj.s in the ensemble.
        step_time       Duration traj.s are advanced each large step.
        history         The complete history of ensemble coordinates.

    More on the history once I implement it.

    """

    def __init__(self, step_time, paving, bin_pop_range, init_trajs,
                 init_time=0.0):
        """
        Create a weighted ensemble of trajectories in phase space.

        Parameters:
            step_time       The duration of a timestep, i.e. the time
                            between periodic resampling pauses.
            paving          The paving to be used to bin the phase
                            space. Must provide a get_bin_num function
                            that, given a set of coordinates in phase
                            space, returns a consistent bin ID (i.e.
                            for any one set of coordinates, it always
                            returns the same ID).
            bin_pop_range   Range of permissible bin traj. counts in the
                            form of a tuple (min, max). If any one bin
                            count is found to be below min and nonzero
                            during the resampling step, the trajectories
                            there are replicated until this number is
                            reached. If a population is above max,
                            traj.s are merged unti max is reached.
            init_trajs      Initial set of (weighted) trajectories to
                            seed the algorithm.

        Optional Parameters:
            init_time       Starting time for the entire ensemble.
                            Ideally all the initial trajectories would
                            also have this value as their init_time, but
                            the ensemble time is tracked separately from
                            each trajectory time, so this is not
                            enforced. Defaults to 0.0.

        """
        self.step_time = step_time
        self.time = init_time
        self.paving = paving
        self.init_trajs = deepcopy(init_trajs)
        self._recompute_bins()
        self.bin_pop_range = bin_pop_range
        self.clone_num = 1

    def run_step(self):
        """
        Run one step of the Weighted Ensemble algorithm.

        A step starts out with a resampling procedure, then runs the
        dynamics of all constituent trajectories for a time
        self.step_time.

        """
        self._resample()
        self._run_dynamics_all()
        self._recompute_bins()
        #self._record_state()

    def run_time(self, duration):
        """
        Run this trajectory until a specified time is reached.

        If the end time is between timesteps, the last step to be run
        will be the latest one ending before the stop time.

        Parameters:
            duration    Amount of time the trajectory will be run. This
                        will be added to the current time to obtain the
                        stop time.

        Returns:
            The ensemble time at the end of the last step.

        """
        stop_time = self.time + duration
        while self.time < stop_time:
            self.run_step()
        return self.time

    def get_pdist(self, paving=None):
        """
        Return the probability distribution describing the system.

        The returned (discrete) distribution is the ensemble's
        approximation to the underlying probability distribution, i.e.
        the analytical solution to the kinetic Master equation in the
        case of a chemical kinetics simulation.

        Parameters:
            paving      The paving defining the regions over which to
                        integrate the probability distribution
                        (analogous to bins of a histogram). If None
                        (the default), the ensemble's internal paving
                        is used.

        Returns:
            One-dimensional array, indexed by bin, giving the value of
            the discrete probability distribution at each bin.

        """
        if paving is not None:
            raise NotImplementedError("No support for arbitrary pavings at " +
                    "this time.")
        self._recompute_bins()
        weights = np.zeros((self.paving.num_bins))
        for bin_id, trjs in self.bins.items():
            weights[bin_id] = sum(trj.weight for trj in trjs)
        return weights

    def _recompute_bins(self):
        """
        Recompute bin numbers for all trajectories.

        Also rebuild the data structure relating bins to trajectories.

        """
        new_bins = defaultdict(lambda: [])
        if hasattr(self, 'bins'):
            trajs = iter(self)
        else:
            trajs = iter(self.init_trajs)
        for traj in trajs:
            #TODO More robust, generalizable way of getting coords from traj.s?
            bin_no = self.paving.get_bin_num(traj.state)
            heappush(new_bins[bin_no], traj)
        self.bins = new_bins

    # TODO Implement history list
    def _record_state(self):
        """
        Record the state in a history list and accumulators.

        """
        raise NotImplementedError()

    def __iter__(self):
        """Iterate over all trajectories in the ensemble."""
        for bin_no, trajs in self.bins.items():
            yield from trajs

    def _run_dynamics_all(self):
        """
        Advance all trajectories in time.

        In principle, this can be done in parallel - for now, though,
        it is done serially.

        """
        for traj in self:
            traj.run_dynamics(self.step_time)
        self.time += self.step_time

    def _resample(self):
        """Resample the phase space by modifying the bin populations."""
        for bin_id, trajs in self.bins.items():
            if len(trajs) > self.bin_pop_range[1]:
                self._reduce_bin(bin_id, trajs, self.bin_pop_range[1])
            elif len(trajs) < self.bin_pop_range[0] and len(trajs) != 0:
                self._grow_bin(bin_id, trajs, self.bin_pop_range[0])

    def _reduce_bin(self, bin_id, trajs, target_pop):
        """
        Combine trajectories to reduce the population of a bin.

        May run into problems if target_pop is less than 2 - however,
        it is assumed this will not be the case.

        """
        while len(trajs) > target_pop:
            traj_minwt = heappop(trajs)
            absorber = heappop(trajs)
            absorber.weight += traj_minwt.weight
            heappush(trajs, absorber)

    def _grow_bin(self, bin_id, trajs, target_pop):
        """Split trajectories to increase the population of a bin."""
        while len(trajs) < target_pop:
            traj_split = max(trajs)
            clones = traj_split.clone(self.clone_num)
            for clone in clones:
                heappush(trajs, clone)


# TODO Make abstract class?
class Paving(object):

    """
    Functionality associated with a paving of phase space.

    Bin indices are assumed to be contiguous, running from 0 up to
    the number of bins defined (minus 1).

    Public methods:
        get_bin_num     Return the bin index for a given state

    Public properties:
        num_bins        The total number of bins in the paving.

    """

    def __init__(self):
        raise NotImplementedError("Abstract class.")

    def __iter__(self):
        raise NotImplementedError("Abstract class.")

    def get_bin_num(self):
        raise NotImplementedError("Abstract class.")


class UniformPaving(Paving):

    """
    A paving of phase space that is uniform in every dimension.

    The paving is delimited by a rectangular region in phase space.
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
        get_bin_num     Return the bin index for a given state.

    Public properties:
        num_bins        The total number of bins in the paving.

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
        if np.any((self.up_bound - self.low_bound) <= 0):
            raise ValueError("Each upper bound must be greater than the " +
                             "corresponding lower bound.")
        self.bin_counts = np.asarray(bin_counts)
        if np.any(self.bin_counts <= 0):
            raise ValueError("Must specify at least one bin in each dimension")
        self.num_bins = np.prod(self.bin_counts)

    # TODO Some bin edges are incorrectly placed by this algorithm. Fix.
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

