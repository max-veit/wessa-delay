#!/usr/bin/python
#
# Copyright © 2014 Max Veit.
#
# This file is part of Max Veit's undergraduate thesis research code.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Some utilities for the thesis project.

Currently contains one utility for manipulating the output of the
weighted-ensemble and SSA algorithms.

No command-line functionality; this module must be imported to be used.

"""

import sys

import numpy as np

import ssad
import ensemble as we

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

