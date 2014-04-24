#!/usr/bin/env python
"""
Usage: runner_ddwe.py param_fname [-o out_fname] [-c]

This script sets up the delayed protein degradation (clock) system,
runs it with the specified parameters, and writes the result to a file.

The parameter 'param_fname' specifies the filename from which to read
the reaction and parameters, 'out_fname' specifies the name of the file
to which to write results (as a NumPy zipped array collection). If '-c'
is specified, the file will be overwritten (clobbered) if it exists.

"""


import sys
import os
import json
import time

import numpy as np
from numpy import random

import ssad
import ensemble as we


def parse_options(args):
    """
    Parse command-line options regarding file input and output.

    Returns method parameters as a (possibly nested) dictionary.  The
    element 'out_fname' contains the name of the file to which to write
    output. The filename specified on the command line overrides any
    specified in the parameter file.

    """
    output_dir_default = 'output'
    out_fname_default = 'delayed_deg_output.npz'
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
            params['out_fname'] = args[fl_idx + 1]
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

def run_ensembles(rxns, ens_params, rxn_params):

    """
    Run several weighted ensembles for this system and gather statistics.

    Parameters:
        rxns        The list of reactions defining the system
        ens_params  Dictionary of ensemble options

    Returns a dict of arrays.
    The array 'bin_xs' is the array of x-positions at which the bins
    start.
    The array 'prob_dists' is an MxN array (N is the number of bins, M
    is the number of ensembles) containing the probability distributions
    generated by each ensemble at the end of the run time.
    The array 'tot_times' contains the total run time for each ensemble.

    """

    binrange = ens_params['binrange']
    bin_xs, bin_width = np.linspace(*binrange,
                                    num=ens_params['nbins'],
                                    endpoint=False,
                                    retstep=True)
    paving = we.UniformPaving(*binrange, bin_counts=ens_params['nbins'])
    prune_itval = ens_params.get('prune_itval')

    prob_dists = np.zeros((ens_params['num_ens'], ens_params['nbins']))
    bin_counts = np.empty((ens_params['num_ens'], ens_params['nbins']))
    tot_times = np.empty((ens_params['num_ens']))
    for ens_idx in range(ens_params['num_ens']):
        # Create seed trajectories with random initial conditions and phases
        init_trjs = []
        for idx in range(ens_params['ntrajs']):
            init_state = random.randint(*binrange)
            init_time = random.random_sample() * rxn_params['tau_delay']
            init_trjs.append(we.WeightedTrajectory(
                [init_state], rxns, 1.0 / ens_params['ntrajs'],
                init_time=init_time))
        ens = we.Ensemble(ens_params['step_time'],
                          paving,
                          ens_params['bin_pop_range'],
                          init_trjs)
        tot_time = ens.run_time(ens_params['tot_time'],
                                prune_itval=prune_itval)
        tot_times[ens_idx] = tot_time
        pdist = ens.get_pdist()
        prob_dists[ens_idx, ...] = pdist
        bin_counts[ens_idx, ...] = ens.get_bin_counts()

    result = {'bin_xs': bin_xs,
              'prob_dists': prob_dists,
              'bin_counts': bin_counts,
              'tot_times': tot_times,
              }
    return result


if __name__ == "__main__":
    params = parse_options(sys.argv)
    rxns = setup_reactions(params['rxn_params'])
    start_time = time.process_time()
    result = run_ensembles(rxns, params['ens_params'], params['rxn_params'])
    run_time = time.process_time() - start_time
    np.savez(params['out_fname'], rxn_params=params['rxn_params'], **result)
    print("Run time: {} seconds.".format(run_time))
