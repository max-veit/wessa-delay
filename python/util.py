#!/usr/bin/python
"""
A general collection of utilities useful for the thesis project.

Contains mostly utilities for manipulating the output of the
weighted-ensemble and SSA algorithms.

No command-line functionality; this module must be imported to be used.

"""

import sys

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

    The average is computed over time, so the value of P(n, m) is the
    total time the trajectory spent at state n while the state tau time
    units ago was m.

    Parameters:
        traj        Trajectory from which to read the history
        tau         Delay time between the two state measurements
        paving_now  Paving for the "current" coordinate
        paving_tau  Paving for the "delayed" coorindate

    Optional Parameters:
        existing    Existing joint probability distribution; this
                    function will add its values into this array.
                    Must be a 2-D array, first index is the "current"
                    coord, second is the "delayed" coord. Obviously, the
                    shape must match the bin counts for each paving.
                    Default None, meaning a new array will be created.
        from_time   Time (on current coordinate) at which to start
                    integrating. Default None, meaning integrate the
                    entire history beginning at t = tau.

    """
    if not from_time:
        from_time = traj.init_time + tau
    if existing is not None:
        jdist = existing
        # TODO Check dimensions?
    else:
        jdist = np.zeros((paving_now.num_bins, paving_tau.num_bins))
    curr_time = from_time
    curr_idx = traj.get_hist_index(from_time)
    delay_idx = traj.get_hist_index(from_time - tau, use_init_state=True)
    while curr_time < traj.time:
        curr_bin = paving_now.get_bin_num(traj.hist_states[curr_idx])
        delay_bin = paving_tau.get_bin_num(traj.hist_states[delay_idx])
        if (curr_idx + 1) < len(traj.hist_times):
            wait_time_curr = traj.hist_times[curr_idx+1] - curr_time
        else:
            wait_time_curr = traj.time - curr_time
        if (delay_idx + 1) < len(traj.hist_times):
            wait_time_delay = traj.hist_times[delay_idx+1] - (curr_time - tau)
        else:
            wait_time_delay = traj.time - (curr_time - tau)
        wait_time = min(wait_time_curr, wait_time_delay)
        jdist[curr_bin][delay_bin] += wait_time
        if wait_time_curr <= wait_time_delay:
            curr_idx += 1
        else:
            delay_idx += 1
        curr_time += wait_time
    return jdist

if __name__ == "__main__":
    print(__doc__)
    sys.exit(1)

