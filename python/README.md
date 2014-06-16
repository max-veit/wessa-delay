#Max Veit's Undergraduate Thesis Project#
##Python implementation##

The implementation of the weighted-ensemble SSA with delays consists of three
Python modules:

    * `ssad.py`
    * `ensemble.py`
    * `util.py`

and two scripts:

    * `runner_wegen.py`
    * `runner_ddjd.py`

The module `ssad` contains the full implementation of the SSA incorporating
delayed reactions. The module `ensemble`, which depends on `ssad`, adds
weighted-ensemble capability. A utility for generating a joint probability
distribution is currently the only function in `util`.

The two scripts are for automated, i.e. non-interactive, execution of the
algorithms for runs that take a long time or use lots of memory. More details
below.

Much of the development and testing of the code, as well as generation,
analysis, and plotting of results, was done using four IPython notebooks:

    * `Testbed.ipynb`, a sandbox notebook used for some of the inital testing
      as well as trying out new features and concepts.

    * `UnitTesting.ipynb` for more systematic testing of specific units of
      functionality (I may replace this with a formal set of tests using the
      `unittest` module at some point, although the stochastic nature of some
      of the algorithms makes precise unit testing difficult).

    * `DelayedDegradation.ipynb`, a detailed investigation of the delayed
      protein degradation system.

    * `Plotting.ipynb` for making nicely formatted plots of results generated
      by the automated scripts.

###Automated runner scripts###

The two scripts with names beginning with `runner` are for automated execution
of the SSA and weighted-ensemble algorithms. They read in parameters from JSON
files and write out the results in NumPy zipped format, so they do not require
an interactive session to use.

The script `runner_wegen.py` is designed to read in an arbitrary set of
reaction pathways and simulate them using the weighted-ensemble SSA. It uses a
uniform paving for the bins used by the weighted-ensemble method. It records
the probability distributions at the end of the dynamics time; multiple
ensembles (with identical parameters) can be run in one job.

The script `runner_ddjd.py` is specialized to the delayed protein degradation
system. It sets up the system, runs it for multiple values of a certain
parameter (the delayed degradation rate $C$), and generates a delayed joint
probability distribution $P(n(t) = p, n(t - \tau) = q)$ for each parameter set.

Parameters for the scripts are stored in JSON files, one parameter set per
file. The individual parameters are not documented, but the files
`params/ddwe_gauss_t1_wa800_res.json` (for use with `runner_ddwe.py`) and
`params/ddjd_csweep3_wa100.json` as well as `params/ddjd_csweep4_wa100.json`
(for `runner_ddjd.py`) provide examples of how these parameters are used. For
the example parameter files included in the `params` directory, the filenames
prefixed `ddwe` and `pdwe` are for use with the script `runner_wegen.py` and
those prefixed `ddjd` are for use with `runner_ddjd.py`. Using a parameter set
with the wrong script will result in various `KeyError`s.

###Parallel Execution###

The codebase is currently entirely serial. The only method of parallelism
available is system-level (i.e. running multiple jobs). I may add
application-level parallelism at some point.
