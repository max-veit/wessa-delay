#!/usr/bin/env python
"""
Usage: delayed_deg_script.py [out_fname] [-f out_fname] [-o]

This script sets up the delayed protein degradation (clock) system,
runs it with the specified parameters, and writes the result to a file.

The parameter 'out_fname' specifies the name of the file to which to
write results (as a NumPy array). If '-o' is specified, the file will
be overwritten if it exists.

"""


import sys

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
    'nbins': 20,
    'binrange': (-0.5, 59.5),
    'step_time': 2.0,
    'run_time': 400.0,
    }

# Command-line options
fname_default = 'delayed_deg_output.npz'
if len(sys.argv) == 1:
    out_fname = fname_default
elif not sys.argv[1].startswith('-'):
    out_fname = sys.argv[1]
elif '-f' in sys.argv:
    fl_idx = sys.argv.index('-f')
    if len(sys.argv) < fl_idx + 2:
        print(__doc__)
        raise RuntimeError("Must specify the output filename with '-f'.")
    else:
        out_fname = sys.argv[fl_idx + 1]
else:
    out_fname = fname_default

if '-o' in sys.argv:
    overwrite = True
else:
    overwrite = False


# Reaction and ensemble setup
rxns = [ssad.Reaction([0], [+1], rxn_params['k_plus']),
        ssad.Reaction([1], [-1], rxn_params['k_minus']),
        ssad.Reaction([1], [-1], rxn_params['k_delayed'],
                      delay=rxn_params['tau_delay'])]
init_trjs = []
binrange = ens_params['binrange']
bin_xs, bin_width = np.linspace(*binrange,
                                num=ens_params[nbins],
                                endpoint=False,
                                retstep=True)
paving = we.UniformPaving(*binrange, bin_counts=ens_params['nbins'])
for idx in range(ens_params['nbins']):
    init_state = random.randint(*binrange)
    init_time = random.random_sample() * rxn_params['tau_delay']
    init_trjs.append(we.WeightedTrajectory(
        [init_state], rxns, 1.0 / ntrajs, init_time=init_time))
