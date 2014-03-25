#!/usr/bin/env python
"""
Usage: delayed_deg_script.py [out_fname] [-f out_fname] [-o]

This script sets up the delayed protein degradation (clock) system,
runs it with the specified parameters, and writes the result to a file.

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


# Adjustable parameters
rxn_params = {
    'k_plus': 100,
    'k_minus': 3,
    'k_delayed': 5,
    'tau_delay': 20,
    }

ens_params = {
    'ntrajs': 40,
    'nbins': 10,
    'binrange': (-0.5, 59.5),
    'step_time': 2.0,
    'tot_time': 100.0,
    'bin_pop_range': (4, 4),
    'resample': True,
    }

def parse_options(args):
    """
    Parse command-line options regarding file output.

    Returns a tuple (fname, overwrite), where fname is the name of the
    file to be written and overwrite is a boolean telling whether to
    overwrite the file if it exists.

    """
    fname_default = 'delayed_deg_output.npy'
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
    return (out_fname, overwrite)


def setup_reactions(rxn_params):
    """Return the system of reactions to be simulated."""
    rxns = [ssad.Reaction([0], [+1], rxn_params['k_plus']),
            ssad.Reaction([1], [-1], rxn_params['k_minus']),
            ssad.Reaction([1], [-1], rxn_params['k_delayed'],
                          delay=rxn_params['tau_delay'])]
    return rxns


def run_ensemble(reactions, ens_params):

    """
    Set up and run the weighted ensemble for this system.

    Parameters:
        reactions   The list of reactions defining the system
        ens_params  Dictionary of ensemble options

    Returns a dict of arrays.
    The element 'prob_dist' is an (M+1) x N array representing the
    probability distribution at each resampling time, where M is the
    number of weighted-ensemble steps and N is the number of bins. The
    first entry (result[0,:]) represents the distribution before any
    trajectories have been run.
    The element 'bin_xs' is the array of x-positions at which the bins
    start.
    The element 'times' is the array of times at which the distribution
    is sampled.

    Note that due to the initial randomization of trajectory phases, the
    distributions at times earlier than tau (the delay time) should not
    be used.

    """

    init_trjs = []
    binrange = ens_params['binrange']
    bin_xs, bin_width = np.linspace(*binrange,
                                    num=ens_params['nbins'],
                                    endpoint=False,
                                    retstep=True)
    paving = we.UniformPaving(*binrange, bin_counts=ens_params['nbins'])

    for idx in range(ens_params['ntrajs']):
        init_state = random.randint(*binrange)
        # Choose a random time between 0 and tau to randomize the phases
        init_time = random.random_sample() * rxn_params['tau_delay']
        init_trjs.append(we.WeightedTrajectory(
            [init_state], rxns, 1.0 / ens_params['ntrajs'],
            init_time=init_time))

    niter = int(ens_params['tot_time'] / ens_params['step_time'])
    prob_dist = np.empty((niter + 1, ens_params['nbins']))
    pdist_times = np.linspace(0, ens_params['tot_time'], niter + 1)
    ens = we.Ensemble(ens_params['step_time'],
                      paving,
                      ens_params['bin_pop_range'],
                      init_trjs)

    for stidx in range(niter):
        prob_dist[stidx, :] = ens.get_pdist()
        ens.run_step(resample=ens_params['resample'])
    prob_dist[niter, :] = ens.get_pdist()
    result = {'prob_dist': prob_dist,
              'bin_xs': bin_xs,
              'times': pdist_times}
    return result


def write_result(result, fname, overwrite):
    if not overwrite and os.path.isfile(fname):
        raise RuntimeError("Error: File " + fname + "exists.\n" +
                           "To overwrite, pass the '-o' option.")
    else:
        np.savez(fname, **result)


if __name__ == "__main__":
    fname, overwrite = parse_options(sys.argv)
    rxns = setup_reactions(rxn_params)
    result = run_ensemble(rxns, ens_params)
    write_result(result, fname, overwrite)
