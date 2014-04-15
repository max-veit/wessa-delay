#!/usr/bin/env python
"""
Usage: dd_joint_pdist.py [out_fname] [-f out_fname] [-o]

This script sets up the delayed protein degradation (clock) system,
runs it with the specified parameters, and computes the delayed joint
probability distribution.

The parameter 'out_fname' specifies the name of the file to which to
write results (as a NumPy zipped array collection). If '-o' is
specified, the file will be overwritten if it exists.

"""

import sys
import os

import numpy as np
from numpy import random

import ssad
import ensemble as we
import util


# Adjustable parameters
rxn_params = {
    'k_plus': 100,
    'k_minus': 3,
    'k_delayed': 3,
    'tau_delay': 20,
    }

sweep_params = {
    'C_min': 1.0,
    'C_max': 5.0,
    'npoints': 9,
    }

pdist_params = {
    'nbins': 60,
    'binrange': (0, 60),
    'init_state': [0],
    'run_time': 120,
    }


def parse_options(args):
    """
    Parse command-line options regarding file output.

    Returns the name of the file to be written.

    """
    fname_default = 'dd_joint_output.npz'
    if len(args) == 1:
        out_fname = fname_default
    elif not args[1].startswith('-'):
        out_fname = args[1]
    elif '-f' in args:
        fl_idx = args.index('-f')
        if len(args) < fl_idx + 2:
            print(__doc__)
            raise RuntimeError("Must specify the output filename with '-f'.")
        else:
            out_fname = args[fl_idx + 1]
    else:
        out_fname = fname_default
    if '-o' in args:
        overwrite = True
    else:
        overwrite = False
    if not overwrite and os.path.isfile(out_fname):
        raise RuntimeError("File " + out_fname + " exists.\n" +
                           "To overwrite, pass the '-o' option.")
    else:
        return out_fname

def setup_reactions(rxn_params):
    """Return the system of reactions to be simulated."""
    rxns = [ssad.Reaction([0], [+1], rxn_params['k_plus']),
            ssad.Reaction([1], [-1], rxn_params['k_minus']),
            ssad.Reaction([1], [-1], rxn_params['k_delayed'],
                          delay=rxn_params['tau_delay'])]
    return rxns

def jdist_sweep(rxn_params, sweep_params, pdist_params):
    """
    Compute the delayed joint probability over a range of parameters.

    Uses the same binning for non-delayed and delayed values.

    Returns a dict of arrays; 'C_vals' is the list of delayed-reaction
    propensity constants that were used. The 3-D array 'j_pdists'
    contains all the calculated distributions, with C-values along the
    first axis, non-delayed bins along the second, and delayed bins
    along the third.

    """
    C_range = np.linspace(sweep_params['C_min'],
                          sweep_params['C_max'],
                          sweep_params['npoints'])
    rxns_sweep = dict(rxn_params)
    nbins = pdist_params['nbins']
    paving = we.UniformPaving(*pdist_params['binrange'], bin_counts=nbins)
    pdists = np.empty((sweep_params['npoints'], nbins, nbins))
    for swidx, C_val in enumerate(C_range):
        rxns_sweep['k_delayed'] = C_val
        rxns = setup_reactions(rxns_sweep)
        trj = ssad.Trajectory(pdist_params['init_state'], rxns)
        trj.run_dynamics(pdist_params['run_time'])
        pdists[swidx,...] = util.delay_joint_pdist(
                trj, rxn_params['tau_delay'], paving, paving)
    return {'C_vals': C_range, 'j_pdists': pdists}

if __name__ == "__main__":
    fname = parse_options(sys.argv)
    result = jdist_sweep(rxn_params, sweep_params, pdist_params)
    np.savez(fname, **result)
