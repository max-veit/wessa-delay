#!/usr/bin/python
"""
A general collection of utilities useful for the thesis project.

Contains mostly utilities for manipulating the output of the
weighted-ensemble and SSA algorithms.

No command-line functionality; this module must be imported to be used.

"""

import sys
import bisect

import numpy as np

import ssad
import ensemble as we

def conditional_delay_average(traj, paving, tau, exclude_begin=True):
    """
    Compute the conditional average <x(t - tau) | x> of a trajectory.

    The average is computed as by tallying x(t - tau) for each
    (x_i, t_i) in the trajectory; the averaging is done over a
    certain range of x_i specified by the paving.

    Parameters:
        traj    The trajectory for which to compute the average; must
                support the same interface as ssad.Trajectory.
        paving  The paving used to subdivide the state space in order to
                compute the average in a discrete way; must expose the
                same interface as ensemble.Paving .
        tau     Delay time used to calculate the average.

    Optional parameters:
        exclude_begin
                Whether to exclude the first tau time units of the
                trajectory in computing the average (in case the values
                of x(t < 0) are not well-defined or do not behave like
                the rest of the trajectory). Default True.

    """
    pass

def delay_joint_pdist(traj, tau, paving_now, paving_tau, existing=None,
        from_time=None):
    """
    Compute the joint probability distribution for a delayed trajectory.

    Parameters:
        traj        Trajectory from which to read the history
        tau         Delay time between the two state measurements
        paving_now  Paving for the "current" coordinate
        paving_tau  Paving for the "delayed" coorindate

    Optional Parameters:
        existing    Existing joint probability distribution; this
                    function will add its tallies into this array.
                    Must be a 2-D array, first index is the "current"
                    coord, second is the "delayed" coord. Obviously, the
                    shape must match the bin counts for each paving.
                    Default None, meaning a new array will be created.
        from_time   Time (on current coordinate) at which to start
                    tallying. Default None, meaning tally the entire
                    history beginning at t = tau.

    """
    if not from_time:
        from_time = traj.init_time + tau
    if existing is not None:
        tallies = existing
        # TODO Check dimensions?
    else:
        tallies = np.zeros((paving_now.num_bins, paving_tau.num_bins))
    start_idx = bisect.bisect_right(traj.hist_times, from_time)
    curr_times = traj.hist_times[start_idx:]
    curr_states = traj.hist_states[start_idx:]
    delay_times = np.array(curr_times) - tau
    delay_states = traj.sample_state_seq(delay_times).transpose()
    for evt_idx in range(len(curr_times)):
        curr_idx = paving_now.get_bin_num(curr_states[evt_idx])
        delay_idx = paving_tau.get_bin_num(delay_states[evt_idx])
        tallies[curr_idx, delay_idx] += 1
    return tallies

if __name__ == "__main__":
    print(__doc__)
    sys.exit(1)

