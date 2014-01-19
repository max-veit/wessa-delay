"""
Stochastic Simulation Algorithm (aka Gillespie algorithm) implementation,
incorporating capability for delayed reactions.

Classes:
    Reaction        Specification for an individual reaction pathway
    Species         Unique identification for a chemical species
    Trajectory      Functionality to evolve a single trajectory in
                    configuration space

"""

import heapq
import functools
from collections import defaultdict
import numpy as np
from numpy.random import exponential, random


"""
Encapsulation of information about a reaction pathway.

Public methods:
    calc_propensity     Given state information, calculate the propensity
                        of this reaction pathway.

"""

class Reaction(object):
    """
    Specify a reaction pathway.

    Parameters:
        reactants       Array of IDs (indices) of the reactants involved
                        in this reaction
        products        Array of indices of the products of this reaction
        state_vec       State-change vector. Integer array of same length
                        as total number of species, which, when added onto
                        an existing state vector, gives the effect of this
                        reaction.
        propensity_const    A constant used in determining the propensity
                        of this reaction. Different meaning for
                        unimolecular, bimolecular, and generation
                        reactions.
    Optional Parameters:
        delay           The delay, if any, associated with this reaction.
                        Default 0.0.

    """
    def __init__(self, reactants, products, state_vec,
                 propensity_const, delay=0.0):
        self.reactants = reactants
        self.products = products
        self.state_vec = state_vec
        self.propensity_const = propensity_const
        self.delay = delay

    # TODO
    # def read_list_from_file(filename):

    """
    Calculate the propensity for this reaction with the given
    concentrations of the reactants. Currently only supports
    unimolecular and bimolecular reactions.

    """
    def calc_propensity(self, state):
        if len(self.reactants) == 0:
            return self.propensity_const
        elif len(self.reactants) == 1:
            return self.propensity_const * state[self.reactants[0]]
        elif len(self.reactants) == 2:
            if self.reactants[0] == self.reactants[1]:
                rct_count = state[self.reactants[0]]
                return (0.5 * rct_count * (rct_count - 1) *
                        self.propensity_const)
            else:
                return (state[self.reactants[0]] *
                        state[self.reactants[1]] *
                        self.propensity_const)
        elif len(reactants) > 2:
            raise NotImplementedError(
                    "Reactions with greater than two reactants are " +
                    "not supported.")

    # TODO Add capability to use species names
    def __str__(self):
        return ("Reaction taking reactants: " + str(self.reactants) +
                " to products: " + str(self.products) +
                " with rate " + str(self.propensity_const) +
                " and delay " + str(self.delay))

"""
A Reaction overridden to always return a constant rate (useful in
biological simulations)

Public methods:
    calc_propensity     Return this reaction's (fixed) rate or propensity.

"""

class ReactionConstRate(Reaction):
    """Return this reaction's (fixed) propensity. State is ignored."""
    def calc_propensity(self, state):
        return self.propensity_const


# TODO Not used - candidate for deletion
# Replace with a dictionary or similar structure relating species to names.
class Species(object):
    def __init__(self, index, species_id):
        self.index = index
        self.species_id = species_id

# TODO Document
# TODO Implement sampling, or more reliable trajectory pausing
class Trajectory(object):
    """
    Initialize a new trajectory.

    Parameters:
        state       Initial state vector. Indices represent species
                    indices, values are the populations of those species.
        weight      Initial statistical weight of this trajectory.
        reactions   List of reaction pathways that govern this trajectory's
                    dynamics.

    Optional Parameters:
        init_time   Simulation time at which this trajectory starts.
                    Defaults to 0.0 time units.

    """
    def __init__(self, state, weight, reactions, init_time=0.0):
        self.state = np.asarray(state)
        self.weight = weight
        self.reactions = reactions
        self.propensities = np.empty((len(reactions)))
        self.time = init_time
        self.rxn_counter = 0
        self.event_queue = []
        self.status_message = "No dynamics run yet."
        self.history = ([self.time], [self.state])

    """
    Run the Gillespie SSA to evolve the initial concentrations in time.

    Parameters:
        duration    The amount of time for which the state should be
                    evolved. The state of the trajectory will reflect
                    that of the system after 'duration' units of time.

    Optional Parameters:
        max_steps   The maximum number of reaction steps that should be
                    run. The trajectory will be evolved until the time runs
                    out or the maximum number of steps is reached,
                    whichever comes first. If the steps limit is reached
                    first, the simulation time will be left at the time of
                    the last reaction successfully executed.
                    If set to None, this parameter will be ignored and the
                    number of steps will be limited only by time.

    """
    def run_dynamics(self, duration, max_steps=None):
        stop_time = self.time + duration
        self.rxn_counter = 0
        while (max_steps is None) or (self.rxn_counter < max_steps - 1):
            # Randomly sample the next reaction as well as its firing time
            for ridx, rxn in enumerate(self.reactions):
                self.propensities[ridx] = rxn.calc_propensity(self.state)
            prop_csum = np.cumsum(self.propensities)
            total_prop = prop_csum[-1]
            wait_time = exponential(1.0 / total_prop)
            next_rxn_time = self.time + wait_time
            rxn_selector = random() * total_prop
            for ridx, rxn in enumerate(self.reactions):
                if prop_csum[ridx] > rxn_selector:
                    next_rxn = rxn
                    break

            # Handle delayed reactions, if any
            execute_delayed = False
            if len(self.event_queue) != 0:
                next_delayed = min(self.event_queue)
                if next_delayed[0] < next_rxn_time:
                    next_delayed = heapq.heappop(self.event_queue)
                    execute_delayed = True

            # Execute the reaction, accounting for delayed reactions
            if execute_delayed:
                self._execute_rxn(*next_delayed)
            else:
                if next_rxn_time > stop_time:
                    self.status_message = "Dynamics time run to completion."
                    break
                if next_rxn.delay > 0.0:
                    heapq.heappush(self.event_queue,
                                   (self.time + next_rxn.delay, next_rxn))
                else:
                    self._execute_rxn(next_rxn_time, next_rxn)

    """Execute a reaction, along with hooks like recording the state."""
    def _execute_rxn(self, time, rxn):
        self.rxn_counter += 1
        self.state = self.state + rxn.state_vec
        self.time = time
        self.history[0].append(self.time)
        self.history[1].append(self.state)

    def __str__(self):
        return ("Trajectory at time " + str(self.time) +
                " . Status message: " + self.status_message +
                "Current state: " + str(self.state))

