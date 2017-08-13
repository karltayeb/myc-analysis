import numpy as np
from mycanalysis.src.ldsmixmethods import (_update_observation_covariance,
                                           _update_transition_covariances,
                                           _update_component_weights,
                                           _update_initial_state_means,
                                           _update_initial_state_covariance,
                                           _estimate_states,
                                           _estimate_responsibilities,
                                           _elbo)


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
        self.transition_matrix = None
        self.transition_covariances = None

        self.observation_matrix = None
        self.observation_covariances = None

        self.initial_state_means = None
        self.initial_state_covariances = None

        self.responsibilities = None

        self.state_means = None
        self.state_covariances = None
        self.pairwise_covariances = None

        self.component_weights = None

        self.K = None  # number of mixture components
        self.U = None  # observation dimensionality
        self.V = None  # state dimensionality
        self.T = None  # timepoints

        self.elbo_history = [-1e100]
        self.expected_log_likelihood_history = [-1e100]
        self.entropy_history = [-1e100]

    def initialize(self, data, K, U, V, T, process_noise, observation_noise):
        """
        Initialize the model
        we need all the model parameters plus initial responsibilities or
        initial states and covariances

        This isn't a very thoughtful initialization but the rest of the code
        should work regardless given this information.
        """
        self.K = K
        self.U = U
        self.V = V
        self.T = T
        N = data.shape[0]

        self.transition_matrix = np.array([[1]], dtype=np.float64)

        transition_covariance = np.array([[process_noise**2]])
        self.transition_covariances = \
            np.array([transition_covariance] * K, dtype=np.float64)

        self.observation_matrix = np.array([[1]], dtype=np.float64)

        self.observation_covariance = \
            np.array([[observation_noise**2]], dtype=np.float64)

        self.initial_state_means = np.zeros((K, V), dtype=np.float64)

        initial_state_covariance = np.array([[1]], dtype=np.float64)
        self.initial_state_covariances = \
            np.array([initial_state_covariance] * K, dtype=np.float64)

        responsibilities = np.random.rand(N, K)
        responsibilities = \
            responsibilities / responsibilities.sum(axis=1)[:, np.newaxis]
        self.responsibilities = responsibilities
        self.component_weights = np.ones(K) / K

        self.state_means = np.zeros((K, T, U))
        self.state_covariances = np.zeros((K, T, V, V))
        self.pairwise_covariances = np.zeros((K, T, V, V))

    def set_initialization(self, dictionary):
        """
        Initialize the model with values from a dictionary
        dictinary entries tat correspond to model attributes will be assigned

        if a necessary parameter setting is not in the dictionary
        it gets set to something uninformitive or completely random
        """
        if 'K' in dictionary:
            self.K = dictionary['K']

        if 'T' in dictionary:
            self.T = dictionary['T']

        if 'U' in dictionary:
            self.U = dictionary['U']

        if 'V' in dictionary:
            self.V = dictionary['V']

        if 'transition_matrix' in dictionary:
            self.transition_matrix = \
                dictionary['transition_matrix'].astype(np.float64)

        if 'transition_covariances' in dictionary:
            self.transition_covariances = \
                dictionary['transition_covariances'].astype(np.float64)

        if 'observation_matrix' in dictionary:
            self.observation_matrix = \
                dictionary['observation_matrix'].astype(np.float64)

        if'observation_covariance' in dictionary:
            self.observation_covariance = \
                dictionary['observation_covariance'].astype(np.float64)

        if 'initial_state_means' in dictionary:
            self.initial_state_means = \
                dictionary['initial_state_means'].astype(np.float64)

            if self.initial_state_means.ndim == 1:
                self.initial_state_means = \
                    np.expand_dims(self.initial_state_means, 1)

        if 'initial_state_covariances' in dictionary:
            self.initial_state_covariances = \
                dictionary['initial_state_covariances'].astype(np.float64)

        if 'responsibilities' in dictionary:
            self.responsibilities = \
                dictionary['responsibilities'].astype(np.float64)

        if 'component_weights' in dictionary:
            self.component_weights = \
                dictionary['component_weights'].astype(np.float64)

        if 'state_means' in dictionary:
            self.state_means = \
                dictionary['state_means'].astype(np.float64)
        else:
            self.state_means = \
                np.zeros((self.K, self.T, self.U), dtype=np.float64)

        if 'state_covariances' in dictionary:
            self.state_covariances = \
                dictionary['state_covariances'].astype(np.float64)
        else:
            self.state_covariances = \
                np.zeros((self.K, self.T, self.V, self.V), dtype=np.float64)

        if 'pairwise_covariances' in dictionary:
            self.pairwise_covariances = \
                dictionary['pairwise_covariances'].astype(np.float64)
        else:
            self.pairwise_covariances = \
                np.zeros((self.K, self.T, self.V, self.V), dtype=np.float64)

    def expectation_maximization(self, data, threshold=1e-3, iter_max=1000):
        """
        perform em
        does variational inference over hidden states in e step
        updates model parameters given hidden state distributions in m step
        """
        elbos = []
        for i in range(iter_max):
            elbos.append(self.estep(data))
            elbos.append(self.mstep(data)[0])
            elbo_diff = elbos[-1] - elbos[-2]
            if elbo_diff < threshold:
                break

    def variational_em(self, data,
                       e_threshold=1e-3, e_itermax=100,
                       em_threshold=1e-8, em_itermax=1000):

        converged = 0
        for i in range(em_itermax):
            # perform variational e step
            self.estep(data, threshold=e_threshold, iter_max=e_itermax)

            self.mstep(data)
            self.elbo(data)

            if self.elbo_delta() < em_threshold:
                if self.elbo_delta() < 0:
                    converged = -1
                else:
                    converged = 1
                break

        return converged

    ########################
    # Maximization methods #
    ########################

    def mstep(self, data):
        """
        mstep updated model parameters:
        component_weights
        observation covariance, shared across clusters
        transition_covariance, unique to each trajectory
        initial state distributions for each trajectory
        """
        K = self.K
        V = self.V

        # compute updated parameters
        component_weights = _update_component_weights(
            self.responsibilities
        )

        observation_covariance = _update_observation_covariance(
            data=data,
            state_means=self.state_means,
            state_covariances=self.state_covariances,
            responsibilities=self.responsibilities
        )

        transition_covariances = np.empty((K, V, V))
        for k in range(K):
            transition_covariances[k] = _update_transition_covariances(
                transition_matrix=self.transition_matrix,
                state_means=self.state_means[k],
                state_covariances=self.state_covariances[k],
                pairwise_covariances=self.pairwise_covariances[k]
            )

        initial_state_means = _update_initial_state_means(
            self.state_means
        )

        initial_state_covariances = np.zeros((K, V, V))
        for k in range(K):
            initial_state_covariances[k] = _update_initial_state_covariance(
                initial_state_mean=self.initial_state_means[k],
                state_means=self.state_means[k],
                state_covariances=self.state_covariances[k],
            )

        # update object attributes with new parameter estimates
        self.component_weights = component_weights
        self.observation_covariance = observation_covariance
        self.transition_covariances = transition_covariances
        self.initial_state_means = initial_state_means
        self.initial_state_covariances = initial_state_covariances

    def update_component_weights(self):
        """
        update component weights
        """
        self.component_weights = _update_component_weights(
                self.responsibilities
            )

    def update_observation_covariance(self, data):
        """
        update observation weights
        """
        self.observation_covariance = _update_observation_covariance(
                data=data,
                state_means=self.state_means,
                state_covariances=self.state_covariances,
                responsibilities=self.responsibilities
            )

    def update_transition_covariances(self):
        """
        update transition covariances for each trajectory
        """
        K = self.K
        V = self.V
        transition_covariances = np.empty((K, V, V))
        for k in range(K):
            transition_covariances[k] = _update_transition_covariances(
                transition_matrix=self.transition_matrix,
                state_means=self.state_means[k],
                state_covariances=self.state_covariances[k],
                pairwise_covariances=self.pairwise_covariances[k]
            )

        self.transition_covariances = transition_covariances

    def update_initial_state_means(self):
        """
        update initial state means
        """
        self.initial_state_means = self.state_means[:, 0]

    def update_initial_state_covariances(self):
        """
        update initial state covariances
        """
        K = self.K
        V = self.V

        initial_state_covariances = np.zeros((K, V, V))
        for k in range(K):
            initial_state_covariances[k] = _update_initial_state_covariance(
                initial_state_mean=self.initial_state_means[k],
                state_means=self.state_means[k],
                state_covariances=self.state_covariances[k],
            )
        self.initial_state_covariances = initial_state_covariances

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

    def _estep(self, data):
        """
        one iteration of variational inference for e step
        """
        self.estimate_responsibilities(data)
        self.estimate_states(data)

    def estimate_responsibilities(self, data):
        """
        estimate posterior assignment probabilities given state estimates
        """
        _estimate_responsibilities(
            data=data,
            state_means=self.state_means,
            state_covariances=self.state_covariances,
            observation_precision=np.linalg.inv(self.observation_covariance),
            component_weights=self.component_weights,
            responsibilities=self.responsibilities
        )

    def estimate_states(self, data):
        """
        estimates state sequences for all trajectories/clusters
        """
        _estimate_states(
            data=data,
            transition_matrix=self.transition_matrix,
            observation_matrix=self.observation_matrix,
            transition_covariances=self.transition_covariances,
            observation_covariance=self.observation_covariance,
            initial_state_means=self.initial_state_means,
            initial_state_covariances=self.initial_state_covariances,
            responsibilities=self.responsibilities,
            state_means=self.state_means,
            state_covariances=self.state_covariances,
            pairwise_covariances=self.pairwise_covariances
        )

    #######################
    # Evaltuation Methods #
    #######################

    def elbo(self, data):
        """
        computed the evidence lower bound of the data
        returns float: evidence lower bound of data
        """
        elbo, expected_likelihood, entropy = _elbo(
            data=data,
            responsibilities=self.responsibilities,
            state_means=self.state_means,
            state_covariances=self.state_covariances,
            pairwise_covariances=self.pairwise_covariances,
            component_weights=self.component_weights,
            observation_covariance=self.observation_covariance,
            initial_state_means=self.initial_state_means,
            initial_state_covariances=self.initial_state_covariances,
            transition_covariances=self.transition_covariances
        )

        self.elbo_history.append(elbo)
        self.expected_log_likelihood_history.append(expected_likelihood)
        self.entropy_history.append(entropy)

        return elbo

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
