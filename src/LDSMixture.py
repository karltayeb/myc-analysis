import numpy as np
from scipy.misc import logsumexp
from src.LinearDynamicalSystem import LinearDynamicalSystem


class LDSMixture:
    def __init__(self):
        """
        transition_matrix:predicts state at time t from state at t-1
        transition_covariance: covariance matrix for state
        observation_matrix: transforms state into observation
        observation_covariance: covariance matrix for observation
        initial_state_means, initial_state_covariances: parameterize
        initial state distribution for each trajectory
        """

        self.filters = []
        self.responsibilities = None
        self.component_weights = None

        self.k_components = None  # number of mixture components
        self.n_dim_obs = None  # observation dimensionality
        self.n_dim_state = None  # state dimensionality
        self.t_timepoints = None  # timepoints

        self.elbo_history = [-1e100]
        self.expected_log_likelihood_history = [-1e100]
        self.entropy_history = [-1e100]

    def initialize(self, data, component_weights=None, responsibilities=None,
                   k_components=1, n_dim_obs=1, n_dim_state=1, t_timepoints=1,
                   initial_state_means=None, initial_state_covariances=None,
                   transition_matrices=None, transition_covariances=None,
                   observation_matrices=None, observation_covariances=None):
        """
        Initialize the model
        we need all the model parameters plus initial responsibilities or
        initial states and covariances

        This isn't a very thoughtful initialization but the rest of the code
        should work regardless given this information.
        """

        # basic assignment of unspecified parameters
        if initial_state_means is None:
            initial_state_means = np.zeros((k_components, n_dim_state))

        if initial_state_covariances is None:
            initial_state_covariances = np.array(
                [np.eye(n_dim_state)] * k_components
            )

        if transition_matrices is None:
            transition_matrices = np.array(
                [np.eye(n_dim_state)] * k_components
            )

        if transition_covariances is None:
            transition_covariances = np.array(
                [np.eye(n_dim_state)] * k_components
            )

        if observation_matrices is None:
            observation_matrices = np.array(
                [np.ones((n_dim_state, n_dim_state))] * k_components
            )

        if observation_covariances is None:
            observation_covariances = np.array(
                [np.eye(n_dim_obs)] * k_components
            )

        if component_weights is None:
            component_weights = np.ones(k_components) / k_components

        if responsibilities is None:
            n_obs = data.shape[0]
            responsibilities = np.random.random((n_obs, k_components))
            responsibilities = \
                responsibilities / responsibilities.sum(axis=1)[:, np.newaxis]

        # initialize filters
        self.filters = []
        for k in range(k_components):
            f = LinearDynamicalSystem(
                 n_dim_state=n_dim_state,
                 n_dim_obs=n_dim_obs,
                 n_obs=1,
                 initial_state_mean=initial_state_means[k],
                 initial_state_covariance=initial_state_covariances[k],
                 transition_matrix=transition_matrices[k],
                 transition_covariance=transition_covariances[k],
                 observation_matrix=observation_matrices[k],
                 observation_covariance=observation_covariances[k],
            )

            self.filters.append(f)

        self.k_components = k_components
        self.n_dim_obs = n_dim_obs
        self.n_dim_state = n_dim_state
        self.t_timepoints = t_timepoints

        self.component_weights = component_weights
        self.responsibilities = responsibilities

    def variational_em(self, data, threshold=1e-8, maxiter=10000):
        pass

    #################################
    # Variational inference methods #
    #################################
    def estimate_responsibilities(self, data):
        """
        estimate posterior assignment probabilities given state estimates
        """
        n_obs = data.shape[0]
        responsibilities = np.zeros((n_obs, self.k_components))
        for k in range(self.k_components):
            responsibilities[:, k] = \
                self.filters[k].expected_log_likelihoods(data).sum(axis=1)

        responsibilities += np.log(self.component_weights)
        responsibilities = np.exp(
            responsibilities -
            logsumexp(responsibilities, axis=1)[:, np.newaxis]
        )

        self.responsibilities = responsibilities

    def estimate_states(self, data):
        """
        estimates state sequences for all trajectories/clusters
        """
        for k in range(self.k_components):
            self.filters[k].smooth_multiple(data, self.responsibilities[:, k])

    def information_estimate_states(self, data):
        """
        estimates state sequences for all trajectories/clusters
        """
        for k in range(self.k_components):
            self.filters[k].information_smooth_multiple(
                data, self.responsibilities[:, k]
            )

    ########################
    # Maximization methods #
    ########################

    def mstep(self, data):
        """
        perform all parameter updestated
        """
        pass

    def update_component_weights(self):
        """
        update component weights
        """
        component_weights = \
            self.responsibilities.sum(axis=0) / self.responsibilities.sum()
        return component_weights

    def update_observation_covariance(self, data, tied=False, inplace=True):
        """
        update observation weights
        """
        observation_covariances = \
            np.zeros((self.k_components, self.n_dim_obs, self.n_dim_obs))

        for k in range(self.k_components):
            observation_covariances[k] = \
                self.filters[k].update_observation_covariance(
                    data, self.responsibilities[:, k], inplace=inplace
                )

        if tied:
            # compute joint observation covariance
            tied_observation_covariance = \
                np.zeros((self.n_dim_obs, self.n_dim_obs))

            for k in range(self.k_components):
                tied_observation_covariance += (
                    self.responsibilities[:, k].sum()
                    * self.filters[k].observation_covariance
                )

            tied_observation_covariance /= data.shape[0]

            for k in range(self.k_components):
                observation_covariances[k] = tied_observation_covariance
                if inplace:
                    self.filters[k].observation_covariance = \
                        tied_observation_covariance

        return observation_covariances

    def update_transition_covariances(self, tied=False, inplace=True):
        """
        update transition covariances for each trajectory
        """
        transition_covariances = \
            np.zeros((self.k_components, self.n_dim_state, self.n_dim_state))

        for k in range(self.k_components):
            transition_covariances[k] = \
                self.filters[k].update_transition_covariance(inplace=inplace)

        if tied:
            pass

        return transition_covariances

    def update_initial_state_means(self, tied=False, inplace=True):
        """
        update initial state means
        """
        initial_state_means = np.zeros((self.k_components, self.n_dim_state))
        for k in range(self.k_components):
            initial_state_means[k] = \
                self.filters[k].update_initial_state_mean(inplace=inplace)

        if tied:
            pass

        return initial_state_means

    def update_initial_state_covariances(self, tied=False, inplace=True):
        """
        update initial state covariances
        """
        initial_state_covariances = \
            np.zeros((self.k_components, self.n_dim_state, self.n_dim_state))

        for k in range(self.k_components):
            initial_state_covariances[k] = self.filters[k].\
                update_initial_state_covariance(inplace=inplace)

        if tied:
            pass

        return initial_state_covariances

    #######################
    # Expectation Methods #
    #######################

    def estep(self, data, threshold=1e-10, iter_max=1000, verbose=False):
        """
        data: samples x timepoints matrix of data
        threshold: covergence threshold for variational inference
        iter_max: maximum number of iterations for variational inference
        show training: if true print change in evidence lower bound at
        each iteration
        """
        result = 0  # didint converge

        for i in range(iter_max):
            self.estimate_states(data)
            self.estimate_responsibilities(data)
            self.elbo(data)

            if verbose:
                print(self.elbo_delta())

            if self.elbo_delta() < threshold:
                if self.elbo_delta() < 0:
                    # not monotonically increasing
                    # hopefully very small/ due to precision
                    result = -1  # elbo decreased
                else:
                    result = 1  # elbo converged

                break

        return result

    #######################
    # Evaltuation Methods #
    #######################

    def elbo(self, data):
        """
        compute evidence lower bound
        """
        elbo = 0

        # expected likelihood of assignments
        elbo += (self.responsibilities * np.log(self.component_weights)).sum()

        # expected likelihood of observations from states
        n_obs = data.shape[0]
        expected_log_lls = np.zeros((n_obs, self.k_components))
        for k, f in enumerate(self.filters):
            expected_log_lls[:, k] = \
                f.expected_log_likelihoods(data).sum(axis=1)

        elbo += (self.responsibilities * expected_log_lls).sum()

        # expected liklihood of states
        for f in self.filters:
            elbo += f.expected_sequence_likelihood()

        # subtract assignment entropy
        elbo -= (self.responsibilities * np.log(self.responsibilities)).sum()

        # expected liklihood of states
        for f in self.filters:
            elbo -= f.sequence_entropy()

        self.elbo_history.append(elbo)
        return elbo

    def log_likelihood(self, data):
        """
        log likelihood of the data given model w/ current state estimates
        """
        log_likelihood = 0

        self.estimate_responsibilities(data)
        n_obs = data.shape[0]
        log_lls = np.zeros((n_obs, self.k_components))
        for k, f in enumerate(self.filters):
            log_lls[:, k] = \
                f.log_likelihoods(data).sum(axis=1)

        log_likelihood = \
            logsumexp(log_lls + np.log(self.component_weights), axis=1).sum()

        return log_likelihood

    def elbo_delta(self):
        delta = 0
        if len(self.elbo_history) > 2:
            delta = self.elbo_history[-1] - self.elbo_history[-2]
        else:
            delta = np.inf

        return delta

    def expected_log_likelihood_delta(self):
        try:
            return self.expected_log_likelihood_history[-1] \
                - self.expected_log_likelihood_history[-2]
        except:
            return np.inf
