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
import numpy as np
from numpy.random import exponential

# TODO Document
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
            return self.propensity_const * state[self.reactants[0])]
        elif len(self.reactants) == 2:
            if self.reactants[0] == self.reactants[1]:
                rct_count = state[self.reactants[0]]
                return (0.5 * rct_count * (rct_count - 1) *
                        self.propensity_const)
            else:
                return (state[self.reactants[0] *
                        state[self.reactants[1] *
                        self.propensity_const)
        elif len(reactants) > 2:
            raise NotImplementedError(
                    "Reactions with greater than two reactants are " +
                    "not supported.")


# TODO Not used - candidate for deletion
class Species(object):
    def __init__(self, index, species_id):
        self.index = index
        self.species_id = species_id

# TODO Document
class Trajectory(object):
    def __init__(self, state, weight, reactions, init_time=0.0):
        self.state = state
        self.weight = weight
        self.reactions = reactions
        self.propensities = np.empty((len(reactions)))
        self.time = init_time
        self.rxn_count = 0
        self.event_queue = []

    """
    Run the Gillespie SSA to evolve the initial concentrations in time.

    Parameters:
        duration    The amount of time for which the state should be
                    evolved. The state of the trajectory will reflect
                    that of the system after 'duration' units of time.

    """
    def run_dynamics(self, duration):
        stop_time = time + duration
        while time <= stop_time:
            # Find the next reaction as well as its firing time
            for ridx, rxn in enumerate(self.reactions):
                self.propensities[ridx] = rxn.calc_propensity(self.state)
            prop_csum = np.cumsum(self.propensities)
            total_prop = prop_csum[-1]
            wait_time = exponential(1.0 / total_prop)
            next_rxn_time = time + wait_time
            rxn_selector = random() * total_prop
            for ridx, rxn in enumerate(self.reactions):
                if prop_csum[ridx] > rxn_selector:
                    next_rxn = rxn
                    break

            # Handle delayed reactions, if any
            if len(self.event_queue) != 0:
                next_delayed = min(self.event_queue)
                if next_delayed[0] < next_rxn_time:
                    next_delayed = heapq.heappop(self.event_queue)
                    self._execute_rxn(*next_delayed)

            # Execute the reaction, accounting for delayed reactions
            if next_rxn_time > stop_time:
                break
            if next_rxn.delay > 0.0:
                heapq.heappush((time + next_rxn.delay, next_rxn))
            else:
                self._execute_rxn(next_rxn_time, next_rxn)

    def _execute_rxn(self, time, rxn):
        self.rxn_counter += 1
        self.state = self.state + rxn.state_vec
        self.time = time

