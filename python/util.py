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

if __name__ == "__main__":
    print __doc__
    sys.exit(1)

