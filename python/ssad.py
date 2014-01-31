"""
Stochastic Simulation Algorithm (aka Gillespie algorithm) implementation,
incorporating capability for delayed reactions.

Classes:
    Reaction        Specification for an individual reaction pathway
    Trajectory      Functionality to evolve a single trajectory in
                    configuration space

"""

import functools
import bisect
import pdb
from collections import defaultdict

import numpy as np
from numpy.random import exponential, random


class Reaction(object):
    """
    Encapsulation of information about a reaction pathway.

    Public methods:
        calc_propensity     Calculate this pathway's propensity.

    """

    def __init__(self, reactants, products, state_vec,
                 propensity_const, delay=0.0):
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
        self.reactants = reactants
        self.products = products
        self.state_vec = np.asarray(state_vec)
        self.propensity_const = propensity_const
        self.delay = delay

    # TODO
    # def read_list_from_file(filename):

    def calc_propensity(self, state):
        """
        Calculate the propensity for this reaction with the given
        concentrations of the reactants. Currently only supports
        unimolecular and bimolecular reactions. Higher-order reactions
        can be expressed as a series of bimolecular reactions.

        Important note: If this reaction is delayed, the caller must
        specify the state as it was at time (t - delay) so that the
        propensity can be calculated correctly.

        """
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


class StepsLimitException(Exception):
    """
    Exception to be raised when the simulation takes more steps than the
    pre-determined limit, if one exists.

    """
    pass


# TODO Replace history sampling with more efficient delayed history trackers
# TODO Consider changing structure of history data
class Trajectory(object):
    """
    A single chemical-kinetic trajectory in state (concentration) space.

    Contains functionality for simulating dynamics of the system, thereby
    extending the trajectory, and for sampling the state of the system
    along the trajectory.

    This class uses the Gillespie Stochastic Simulation Algorithm (SSA) to
    simulate the time evolution of a small chemical system.

    Public methods:
        run_dynamics        Run the SSA to advance the trajectory in time.
        sample_state        Sample the system's state at a given time.
        sample_state_seq    Sample the state at a sequence of times.

    """

    def __init__(self, state, reactions, weight=1, init_time=0.0):
        """
        Initialize a new trajectory.

        Parameters:
            state       Initial state vector. Indices represent species
                        indices, values are the populations of those species.
            reactions   List of reaction pathways that govern this trajectory's
                        dynamics.

        Optional Parameters:
            weight      Initial statistical weight of this trajectory, useful
                        in ensemble methods. Defaults to 1.
            init_time   Simulation time at which this trajectory starts.
                        Defaults to 0.0 time units.

        """
        self.state = np.asarray(state)
        self.weight = weight
        self.reactions = reactions
        self.init_time = init_time
        self.time = init_time
        self.rxn_counter = 0
        self.hist_times= [self.time]
        self.hist_states = [self.state]
        self.next_rxn = None
        self.next_rxn_time = None
        self.rxn_tallies = defaultdict(lambda: 0)
        self.reject_tallies = defaultdict(lambda: 0)

    # TODO Systematically test restart capability
    # TODO Rethink storing of next reaction times - isn't the exponential
    #      distribution memoryless? Impact on weighted-ensemble methods?
    #      Delayed reactions (non-Markovian)??
    def run_dynamics(self, duration, max_steps=None):
        """
        Run the Gillespie SSA to evolve the initial concentrations in time.

        If this method has been run on an object before, subsequent runs will
        pick up where the last run left off. This means the effect of:
            trj1.run_dynamics(duration1)
            trj1.run_dynamics(duration2)
        will be the same as that of
            trj1.run_dynamics(duration1 + duration2)
        assuming the pseudorandom number generator's state is not changed
        in between the two calls.
        This capability is useful in, for example, weighted-ensemble
        methods where the ability to pause and restart trajectories without
        introducing any statistical bias is necessary.

        Parameters:
            duration    The amount of time for which the state should be
                        evolved. The state of the trajectory will reflect
                        that of the system after 'duration' units of time.

        Optional Parameters:
            max_steps   The maximum number of steps (reactions) that should be
                        run. The trajectory will be evolved until the time runs
                        out or the maximum number of steps is reached,
                        whichever comes first. If the steps limit is reached
                        first, the simulation time will be left at the time of
                        the last reaction successfully executed and an
                        exception will be raised.
                        If set to None, this parameter will be ignored and the
                        number of steps will be limited only by time.

        """
        stop_time = self.time + duration
        if max_steps is not None:
            stop_steps = self.rxn_counter + max_steps
        if (self.next_rxn is not None) and (self.next_rxn_time is not None):
            resume = True
        else:
            resume = False

        while self.time < stop_time:
            if resume:
                next_rxn = self.next_rxn
                next_rxn_time = self.next_rxn_time
                resume = False
            else:
                next_rxn, wait_time = self._sample_next_reaction()
                next_rxn_time = self.time + wait_time

            # Execute the reaction, if possible
            if self._can_run_rxn(next_rxn):
                if next_rxn_time > stop_time:
                    self._save_run_state(next_rxn, next_rxn_time)
                    self.time = stop_time
                    break
                else:
                    self._execute_rxn(next_rxn, next_rxn_time)
            # If the reaction can't run, wait until the next cycle and
            # select another.
            else:
                self.reject_tallies[next_rxn] += 1

            # This may not stop the trajectory _exactly_ at the limit, but
            # that's not a big problem right now.
            if max_steps is not None:
                if self.rxn_counter >= stop_steps:
                    raise StepsLimitException(
                        "Trajectory run reached maximum allowed number " +
                        "of steps (limit was " + str(max_steps) + ").")

    def _can_run_rxn(self, rxn):
        """
        Determine whether a reaction is allowed to run.

        Uses state information to determine whether a given reaction would
        cause any element of the state vector to go negative.

        Parameters:
            rxn     Reaction to check

        Returns:
            Whether the reaction should be allowed to run, as a boolean.

        """
        return np.all(self.state + rxn.state_vec >= 0)

    def _sample_next_reaction(self):
        """
        Use random sampling to select the next reaction and firing time.

        Returns a tuple of (reaction, wait time). The wait time is the duration
        from the current simulation time to the time the next reaction fires.
        It is sampled from an exponential distribution.

        Handles delayed reactions by looking back in the history and using the
        state as it was (delay) time units ago.

        """
        #pdb.set_trace()
        propensities = np.empty((len(self.reactions)))
        for ridx, rxn in enumerate(self.reactions):
            if rxn.delay == 0.0:
                state = self.state
            else:
                state = self.sample_state(
                    self.time - rxn.delay, use_init_state = True)
            propensities[ridx] = rxn.calc_propensity(state)
        prop_csum = np.cumsum(propensities)
        total_prop = prop_csum[-1]
        wait_time = exponential(1.0 / total_prop)
        rxn_selector = random() * total_prop
        for ridx, rxn in enumerate(self.reactions):
            if prop_csum[ridx] >= rxn_selector:
                next_rxn = rxn
                break
        return (next_rxn, wait_time)

    def _execute_rxn(self, rxn, time):
        """Execute a reaction, along with hooks like recording the state."""
        self.rxn_counter += 1
        self.state = self.state + rxn.state_vec
        self.time = time
        self.hist_times.append(self.time)
        self.hist_states.append(self.state)
        self.rxn_tallies[rxn] += 1

    def _save_run_state(self, rxn, time):
        """Save the trajectory state before pausing."""
        self.next_rxn = rxn
        self.next_rxn_time = time

    def sample_state(self, time, use_init_state=False):
        """
        Return a single sample of this trajectory's state at a given time.

        Parameters:
            time    Any time in the range (self.start_time, self.time)
                    at which to draw the sample.

        Optional Parameters:
            use_init_state  Whether to return this trajectory's initial state
                            if time is earlier than this trajectory's starting
                            time; if this option is set to False, that
                            condition will raise an exception.
                            Default False.

        Returns:
            State vector at the given time, as a 1-D NumPy array.

        Note: If a large number of samples is desired, the function
        sample_state_seq will likely be more efficient.

        """
        if (time > self.time) or (time < self.init_time and
                                  not use_init_state):
            raise ValueError("Out-of-bounds time " + str(time) + " received.")
        if (time < self.init_time) and use_init_state:
            return self.hist_states[0]
        idx = bisect.bisect_right(self.hist_times, time)
        if idx > 0:
            return (self.hist_states)[idx - 1]

    # TODO Implement use_init_state capability (like in sample_state)
    def sample_state_seq(self, times):
        """
        Return a set of samples of the system state.

        Parameters:
            times   A sequence of times at which to sample. The sequence
                    must be sorted in increasing order and not contain any
                    times that are earlier than the trajectory's
                    starting time or later than the trajectory's
                    current time.

        Returns:
            A 2-D NumPy array containing the state vectors at the specified
            sample times. Species are along the first axis, times are along
            the second.

        """
        if not all(times[i] <= times[i+1] for i in range(len(times) - 1)):
            raise ValueError("The sequence of times is not sorted.")
        if (times[-1] > self.time):
            raise ValueError("Latest time " + str(times[-1]) + " is later " +
                             "than the current trajectory time.")
        if (times[0] < self.init_time):
            raise ValueError("First time " + str(times[0]) + " is earlier " +
                             "than the trajectory's starting time.")
        states = np.empty((self.state.size, times.size))
        hist_times = iter(self.hist_times)
        hist_states = iter(self.hist_states)
        # Don't do exception handling - each history has at least one element
        curr_time = next(hist_times)
        curr_state = next(hist_states)
        prev_state = curr_state
        for tidx in range(times.size):
            while curr_time < times[tidx]:
                prev_state = curr_state
                try:
                    curr_state = next(hist_states)
                    curr_time = next(hist_times)
                except StopIteration:
                    break
            states[:,tidx] = prev_state
        return states

    # TODO Consider functionality for selective omission of history
    def clone(self, num_clones, weights=None):
        """
        Copy this trajectory to obtain num_clones _total_ trajectories.

        This method creates copies identical to this trajectory, in the
        sense that the copies have identical history up to the current
        trajectory time.

        Parameters:
            num_clones  The _total_ number of identical trajectories to
                        produce (including this one!).

        Optional Parameters:
            weights     The weights to assign to the group of
                        trajectories. May be either a number, a list of
                        length num_clones, or None.
                        If None, each trajectory (including this one) is
                        assigned the weight self.weight / num_clones.
                        If a number, each trajectory is assigned that
                        number as a weight.
                        If a list, each trajectory will be assigned a
                        unique element of the list as a weight.
                        Default None.

        Returns:
            A list of trajectories of length num_clones. If weights was
            specified as a list, the weights of the output trajectories
            will be in the same order as that list.

        """
        if num_clones < 1:
            raise ValueError("Must specify a positive number of clones.")
        clones = []
        if not np.isscalar(weight) and np.asarray(weight).size != num_clones:
            raise ValueError("Weight list must be of the same size as the " +
                             "number of clones.")
        for cidx in range(num_clones):
            new_clone = Trajectory(self.hist_states[0], self.rxns,
                                   init_time=self.hist_times[0])
            # A deep copy is probably not necessary here, as the past
            # history should not be modified.
            new_clone.hist_times = list(self.hist_times)
            new_clone.hist_states = list(self.hist_states)
            if weight is None:
                new_clone.weight = self.weight / num_clones
            elif np.isscalar(weight):
                new_clone.weight = weight
            else:
                new_clone.weight = weight[cidx]
            clones.append(new_clone)
        return clones

    def __str__(self):
        msg = ("Trajectory at time " + str(self.time) +
               " . Current state: " + str(self.state) + ".")
        if self.next_rxn_time is not None:
            msg += (" Next reaction scheduled for time " +
                    str(self.next_rxn_time) + ".")
        return msg

if __name__ == "__main__":
    raise NotImplemented("No script functionality here.")
