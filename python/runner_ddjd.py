#!/usr/bin/env python
"""
Usage: runner_ddjd.py param_fname [-o out_fname] [-c]

This script sets up the delayed protein degradation (clock) system,
runs it with the specified parameters, and computes the delayed joint
probability distribution.

The parameter 'param_fname' specifies the filename from which to read
the reaction and parameters, 'out_fname' specifies the name of the file
to which to write results (as a NumPy zipped array collection). If '-c'
is specified, the file will be overwritten (clobbered) if it exists.

"""

import sys
import os
import json

import numpy as np
from numpy import random

import ssad
import ensemble as we
import util


def parse_options(args):
    """
    Parse command-line options regarding file input and output.

    Returns method parameters as a (possibly nested) dictionary.  The
    element 'out_fname' contains the name of the file to which to write
    output. The filename specified on the command line overrides any
    specified in the parameter file.

    """
    output_dir_default = 'output'
    out_fname_default = 'dd_joint_output.npz'
    params = dict()
    if len(args) == 1 or args[1].startswith('-'):
        raise RuntimeError("Must specify a filename for parameters.")
    else:
        param_fname = args[1]
    with open(param_fname) as param_file:
        params = json.load(param_file)
    if '-o' in args:
        fl_idx = args.index('-o')
        if len(args) < fl_idx + 2:
            print(__doc__)
            raise RuntimeError("Must specify the output filename with '-o'.")
        else:
            out_fname = args[fl_idx + 1]
    elif 'out_fname' not in params:
        # Construct a default filename
        if os.path.isdir(output_dir_default):
            out_basename = os.path.basename(param_fname)
            out_base = os.path.splitext(out_basename)[0] + ".npz"
            out_fname = os.path.join(output_dir_default, out_base)
        else:
            out_fname = out_fname_default
    params['out_fname'] = out_fname
    if '-c' in args:
        overwrite = True
    else:
        overwrite = False
    if not overwrite and os.path.isfile(out_fname):
        raise RuntimeError("File " + out_fname + " exists.\n" +
                           "To overwrite, pass the '-c' option.")
    else:
        return params

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
        prune_itval = pdist_params.get('prune_itval', None)
        # Run the trajectory, pruning history if necessary
        if prune_itval is not None:
            stop_time = pdist_params['run_time']
            last_read_time = rxn_params['tau_delay']
            pdists[swidx,...] = 0.0
            while trj.time < stop_time:
                trj.run_dynamics(duration=prune_itval)
                pdists[swidx,...] += util.delay_joint_pdist(
                        trj, rxn_params['tau_delay'], paving, paving,
                        from_time=last_read_time)
                last_read_time = trj.time
                trj.prune_history()
            trj.run_dynamics(duration=None, stop_time=stop_time)
            pdists[swidx,...] += util.delay_joint_pdist(
                    trj, rxn_params['tau_delay'], paving, paving,
                    from_time=last_read_time)
        else:
            trj.run_dynamics(pdist_params['run_time'])
            pdists[swidx,...] = util.delay_joint_pdist(
                    trj, rxn_params['tau_delay'], paving, paving)
    return {'C_vals': C_range, 'j_pdists': pdists}

if __name__ == "__main__":
    params = parse_options(sys.argv)
    result = jdist_sweep(params['rxn_params'], params['sweep_params'],
                         params['pdist_params'])
    np.savez(params['out_fname'], **result)
