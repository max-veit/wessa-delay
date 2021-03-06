#!/usr/bin/python3
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
Usage: runner_wegen.py param_fname [-o out_fname] [-c]

This script sets up a system with the reactions specified in the input
file, runs multiple weighted ensembles, and writes the result to a file.

The parameter 'param_fname' specifies the filename from which to read
the reaction and parameters, 'out_fname' specifies the name of the file
to which to write results (as a NumPy zipped array collection). If '-c'
is specified, the file will be overwritten (clobbered) if it exists.

If no output filename is specified, the output will be written to the
file '<basename>.npz' in the directory 'output', which is assumed to
exist in the current directory. The '<basename>' is the base of the
name of the parameter file (i.e. with the '.json' extension stripped).

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
    params['param_fname'] = param_fname
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

def setup_reactions(species, rxn_list):
    """
    Parse the list of reactions to be simulated.

    Return a list of corresponding Reaction objects.

    The parameter 'species' should be a list of the names of species
    present in the system. Species indices will be assigned based on
    their positions in this list.

    The parameter 'rxn_list' should be an iterable containing a reaction
    specification for each reaction in the system. A reaction
    specification should be a mapping containing elements: 'reactants',
    itself a mapping from species name to the number of molecules of
    that species required to run that reaction; 'state_vec', a mapping
    from species name to the amount by which the reaction changes the
    population of that species; and 'propensity_const', the constant
    used to calculate the reaction's propensity as a function of
    reactant concentrations. If the reaction specification contains an
    element 'delay', that number will be used as the reaction's time
    delay (otherwise the reaction will be taken to be instantaneous).

    """
    species_idces = dict()
    for species_idx, species_name in enumerate(species):
        species_idces[species_name] = species_idx
    rxns = []
    for rxn_spec in rxn_list:
        reactants = np.zeros((len(species)))
        for species_name, rct_num in rxn_spec['reactants'].items():
            reactants[species_idces[species_name]] = rct_num
        state_vec = np.zeros((len(species)))
        for species_name, state_num in rxn_spec['state_vec'].items():
            state_vec[species_idces[species_name]] = state_num
        delay = rxn_spec.get('delay', 0.0)
        rxns.append(ssad.Reaction(reactants, state_vec,
                                  rxn_spec['propensity_const'], delay=delay))
    return rxns

def run_ensembles(rxns, ens_params):

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
            init_time = random.random_sample() * ens_params['phase_rand_time']
            init_trjs.append(we.WeightedTrajectory(
                [init_state], rxns, 1.0 / ens_params['ntrajs'],
                init_time=init_time))
        ens = we.Ensemble(ens_params['step_time'],
                          paving,
                          ens_params['bin_pop_range'],
                          init_trjs)
        tot_time = ens.run_time(ens_params['tot_time'],
                                prune_itval=prune_itval,
                                resample=ens_params['resample'])
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
    rxns = setup_reactions(params['species'], params['rxn_list'])
    start_time = time.process_time()
    result = run_ensembles(rxns, params['ens_params'])
    run_time = time.process_time() - start_time
    np.savez(params['out_fname'], param_fname=params['param_fname'], **result)
    print("\nRun time: {:.3g} minutes.".format(run_time / 60.0))
