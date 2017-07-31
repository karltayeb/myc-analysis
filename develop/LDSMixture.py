from pykalman import KalmanFilter
import pykalman.standard as filtermethods
import numpy as np
import scipy as sp
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


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

        # evidence lower bound = expected complete log likelihood + entropy
        self.lower_bound = None

        # expected complete log likelihood E[logP(X, Y, Z | parameters)]
        self.expected_likelihoods = None

        # negative expected log likelihood of latent variable distributions
        self.entropies = None

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

        self.transition_matrix = np.array([[1]])
        transition_covariance = np.array([[process_noise**2]])
        self.transition_covariances = np.array([transition_covariance] * K)

        self.observation_matrix = np.array([[1]])
        self.observation_covariance = np.array([[observation_noise**2]])

        self.initial_state_means = np.zeros(K)
        self.initial_state_covariance = np.array([[1]])

        self.initial_state_means = np.zeros(K)
        initial_state_covariance = np.array([[1]])
        self.initial_state_covariances = \
            np.array([initial_state_covariance] * K)

        responsibilities = np.random.rand(N, K)
        responsibilities = \
            responsibilities / responsibilities.sum(axis=1)[:, np.newaxis]
        self.responsibilities = responsibilities

        self.component_weights = np.ones(K) / K

        self.state_means = np.empty((K, T, U))
        self.state_covariances = np.empty((K, T, V, V))
        self.pairwise_covariances = np.empty((K, T, V, V))

    def em(self, data, threshold=1e-3, iter_max=1000):
        """
        perform em
        does variational inference over hidden states in e step
        updates model parameters given hidden state distributions in m step
        """
        elbos = []
        for i in range(iter_max):
            elbos.extend(self.estep(data))
            elbos.append(self.mstep(data)[0])
            elbo_diff = elbos[-1] - elbos[-2]
            if elbo_diff < threshold:
                break

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
            responsibilities=self.responsibilities
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
            state_means=self.state_means
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

        return self.elbo(data)

    def update_component_weights(self):
        """
        update component weights
        """
        self.component_weights = _update_component_weights(
                responsibilities=self.responsibilities
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

    def estep(self, data, threshold=1e-5, iter_max=1000, show_training=False):
        """
        data: samples x timepoints matrix of data
        threshold: covergence threshold for variational inference
        iter_max: maximum number of iterations for variational inference
        show training: if true print change in evidence lower bound at
        each iteration
        """
        elbos = []

        for i in range(iter_max):
            self._estimate_states(data)
            self._estimate_responsibilities(data)
            elbos.append(self.elbo(data))

            if i >= 1:
                elbo_diff = elbos[-1] - elbos[-2]
                if show_training:
                    print(elbo_diff)
                assert(elbo_diff >= 0)
                if elbo_diff < threshold:
                    break

        return elbos

    def _estep(self, data):
        """
        one iteration of variational inference for e step
        """
        self._estimate_responsibilities(data)
        self._estimate_states(data)

        return self.elbo(data)

    def _estimate_responsibilities(self, data):
        """
        estimate posterior assignment probabilities given state estimates
        """
        arguments = {
            'state_means': self.state_means,
            'state_covariances': self.state_covariances,
            'observation_covariance': self.observation_covariance,
            'component_weights': self.component_weights
        }

        expected_conditional_likelihoods = np.array(list(map(
            lambda obs: _expected_conditional_likelihoods(obs, **arguments),
            data
        )))

        responsibilities = np.array(list(map(
            _responsibilities_update, expected_conditional_likelihoods
        )))

        self.responsibilities = responsibilities

    def _filter_and_smooth(self, f, included, data):
        """
        f: kalman filter object
        included: boolean array indicating which
        data points have non-zero assignment probability
        data: data to estimate states on

        kalman filtering step, estimates distribution over state sequence
        given all of the data. relies on kalman filter package pykalman
        """
        # estimate states
        if f is not None:
            # f is none when no observations areassigned to it
            Z = f._parse_observations(data[included].T)

            (transition_matrices, transition_offsets,
             transition_covariance, observation_matrices,
             observation_offsets, observation_covariance,
             initial_state_mean, initial_state_covariance) = (
                f._initialize_parameters()
            )

            (predicted_state_means, predicted_state_covariances,
             _, filtered_state_means, filtered_state_covariances) = (
                filtermethods._filter(
                    transition_matrices, observation_matrices,
                    transition_covariance, observation_covariance,
                    transition_offsets, observation_offsets,
                    initial_state_mean, initial_state_covariance, Z
                )
            )

            (smoothed_state_means, smoothed_state_covariances,
                kalman_smoothing_gains) = (
                filtermethods._smooth(
                    transition_matrices, filtered_state_means,
                    filtered_state_covariances, predicted_state_means,
                    predicted_state_covariances
                )
            )

            pairwise_covariances = filtermethods._smooth_pair(
                smoothed_state_covariances,
                kalman_smoothing_gains
            )

            state_means = smoothed_state_means
            state_covariances = smoothed_state_covariances
            pairwise_covariances = pairwise_covariances

        else:
            # no observations are assigned, this cluster isn't being used
            state_means = None
            state_covariances = None
            pairwise_covariances = None

        return state_means, state_covariances, pairwise_covariances

    def _estimate_states(self, data):
        """
        estimates state sequences for all trajectories/clusters
        """
        K = self.K

        # set up filters
        filters = [
            _initialize_filter(
                transition_matrix=self.transition_matrix,
                observation_matrix=self.observation_matrix,
                transition_covariance=self.transition_covariances[k],
                observation_covariance=self.observation_covariance,
                initial_state_mean=self.initial_state_means[k],
                initial_state_covariance=self.initial_state_covariance,
                responsibilities=self.responsibilities[:, k]
            )
            for k in range(K)
        ]

        # estimate state distributions
        for k in range(K):
            f, included = filters[k]
            means, covariances, pairwise_covariances = \
                self._filter_and_smooth(f, included, data)

            if means is not None:
                # if we actually got new state estimates update
                # model attributes
                self.state_means[k] = means
                self.state_covariances[k] = covariances
                self.pairwise_covariances[k] = pairwise_covariances

    def elbo(self, data):
        """
        computed the evidence lower bound of the data
        returns float: evidence lower bound of data
        """
        K = self.K
        N = data.shape[0]  # number of observations

        entropies = np.zeros(N + K)
        expected_likelihoods = np.zeros(N + K)

        for j, observation in enumerate(data):
            expected_likelihoods[j] = _expected_observation_likelihood(
                observation=observation,
                responsibilities=self.responsibilities[j],
                state_means=self.state_means,
                state_covariances=self.state_covariances,
                observation_covariance=self.observation_covariance,
                component_weights=self.component_weights
                )

        for j, observation in enumerate(data):
            entropies[j] = \
                _assignment_entropy(responsibilities=self.responsibilities[j])

        for k in range(K):
            expected_likelihoods[N + k] = _expected_sequence_likelihood(
                initial_state_mean=self.initial_state_means[k],
                initial_state_covariance=self.initial_state_covariances[k],
                transition_covariance=self.transition_covariances[k],
                state_means=self.state_means[k],
                state_covariances=self.state_covariances[k],
                pairwise_covariances=self.pairwise_covariances[k]
            )

            entropies[N + k] = _state_sequence_entropy(
                state_covariances=self.state_covariances[k],
                pairwise_covariances=self.pairwise_covariances[k]
            )

        seperate_elbos = expected_likelihoods + entropies
        elbo = seperate_elbos.sum()

        self.lower_bound = elbo
        self.expected_likelihoods = expected_likelihoods
        self.entropies = entropies

        return elbo


def _initialize_filter(transition_matrix, observation_matrix,
                       transition_covariance, observation_covariance,
                       initial_state_mean, initial_state_covariance,
                       responsibilities):
    """
    creates kalman filter object from model parameters and responsibilities
    if responsibilities are zero for any observation it will not be included
    to avoid infinite varaince in the observation covariance matrix for all
    observation.

    each filter object has its corresponding model paramters and a large block
    diagonal matrix of the observation covariance scaled by the responsibility

    returns a filter object and a boolean array of included observations
    """

    included = np.logical_not(np.isclose(responsibilities, 0))
    observation_dim = np.sum(included)

    if observation_dim > 0:
        block_observation_covariance = sp.linalg.block_diag(
            *[observation_covariance / r for r in responsibilities[included]]
            )

        f = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=np.tile(
                observation_matrix, (observation_dim, 1)
                ),
            transition_covariance=transition_covariance,
            observation_covariance=block_observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            n_dim_state=1, n_dim_obs=observation_dim
        )
    else:
        f = None

    return f, included


"""
PARAMETER UPDATES
"""


def _responsibilities_update(expected_conditional_likelihoods):
    responsibilities = np.exp(
            expected_conditional_likelihoods -
            logsumexp(expected_conditional_likelihoods)
    )

    return responsibilities


def _expected_squared_error(observation, state_means, state_covariances,
                            observation_precision, observation_matrix):

    """
    (y-x)T R-1 (y-x)
    """
    T = observation.shape[0]
    expected_squared_error = 0
    for t in range(T):
        residual = (observation[t] -
                    np.dot(observation_matrix, state_means[t])
                    ).reshape(-1, 1)
        expected_squared_error += np.linalg.multi_dot([
            residual.T,
            observation_precision,
            residual
        ])

    expected_squared_error += np.linalg.multi_dot([
        observation_precision,
        observation_matrix,
        state_covariances.sum(axis=0),
        observation_matrix
    ])

    return expected_squared_error


def _expected_squared_errors(data, state_means, state_covariances,
                             observation_covariance, observation_matrix):
    N = data.shape[0]
    K = state_means.shape[0]
    expected_squared_errors = np.empty((N, K))
    observation_precision = np.linalg.pinv(observation_covariance)
    for j, observation in enumerate(data):
        for k in range(K):
            expected_squared_errors[j, k] = _expected_squared_error(
                observation=observation,
                state_means=state_means[k],
                state_covariances=state_covariances[k],
                observation_precision=observation_precision,
                observation_matrix=observation_matrix
            )
    return expected_squared_errors


"""
EXPECTED LIKELIHOOD AND ENTROPY STUFF
"""


def _expected_observation_likelihood(observation, responsibilities,
                                     state_means, state_covariances,
                                     observation_covariance,
                                     component_weights):
    """
    expected likelihood of a single observation sequence
    expectation taken over posterior distribution of assignment
    and state sequences
    """
    expected_conditional_likelihoods = _expected_conditional_likelihoods(
        observation, state_means, state_covariances,
        observation_covariance, component_weights
    )

    expected_observation_likelihood = np.sum(
        responsibilities * expected_conditional_likelihoods
    )

    return expected_observation_likelihood


def _expected_conditional_likelihoods(observation, state_means,
                                      state_covariances,
                                      observation_covariance,
                                      component_weights):
    """
    expected likelihood of an observation given a component
    returns a K vector where each is the conditional expected
    likelihood that the observation was generated from that component
    note this includes component mixture probability
    """

    T = observation.shape[0]
    K = component_weights.shape[0]

    conditional_expected_likelihoods = np.zeros(K)

    observation_precision = np.linalg.pinv(observation_covariance)
    for k in range(K):
        sub = 0
        for t in range(T):
            sub += _expected_normal_logpdf(
                x=observation[t],
                mean=state_means[k, t],
                covariance=observation_covariance,
                precision=observation_precision,
                mean_covariance=state_covariances[k, t]
            )

        conditional_expected_likelihoods[k] = \
            np.log(component_weights[k]) + sub

    return conditional_expected_likelihoods


def _expected_sequence_likelihood(initial_state_mean, initial_state_covariance,
                                  transition_covariance, state_means,
                                  state_covariances, pairwise_covariances):

    """
    expectation of state sequence likelihood
    expectation taken over estimated state distribution
    """
    expected_sequence_likelihood = 0
    expected_sequence_likelihood += _expected_normal_logpdf(
        x=state_means[0],
        mean=initial_state_mean,
        covariance=initial_state_covariance,
        mean_covariance=state_covariances[0]
    )

    T = state_means.shape[0]
    for t in range(1, T):
        covariance = state_covariances[t] + state_covariances[t-1] - \
            (2 * pairwise_covariances[t])

        expected_sequence_likelihood += _expected_normal_logpdf(
            x=state_means[t],
            mean=state_means[t-1],
            covariance=transition_covariance,
            mean_covariance=covariance
        )

    return expected_sequence_likelihood


def _assignment_entropy(responsibilities):
    """
    entropy of posterior assignment estimate
    """
    active_responsibilities = responsibilities[
        np.logical_not(np.isclose(responsibilities, 0))]
    assignment_entropy = -1 * \
        (active_responsibilities * np.log(active_responsibilities)).sum()
    return assignment_entropy


def _state_sequence_entropy(state_covariances, pairwise_covariances):
    """
    entropy of posterior state sequence estimate
    state_covariances: T x V x V covariance matrices for state estimates
    pairwise_covariances: T x V x V
    t-th entry covariance matrix for states at time t, t-1
    """
    T = state_covariances.shape[0]
    state_sequence_entropy = 0
    state_sequence_entropy += \
        multivariate_normal.entropy(cov=state_covariances[0])

    for t in range(1, T):
        # covariance of state at time t given state at time t-1
        covariance = state_covariances[t] - np.linalg.multi_dot([
                pairwise_covariances[t],
                state_covariances[t-1],
                pairwise_covariances[t]
            ])
        state_sequence_entropy += multivariate_normal.entropy(cov=covariance)

    return state_sequence_entropy


def _update_observation_covariance(data, state_means, state_covariances,
                                   responsibilities):
    """
    observation covariance update for all filters
    """
    N, T = data.shape[:2]
    K = responsibilities.shape[1]

    observation_covariance = 0
    for n in range(N):
        for k in range(K):
            for t in range(T):
                observation = data[n, t].reshape(-1, 1)
                state = state_means[k, t].reshape(-1, 1)
                yy = np.dot(observation, observation.T)
                xx = np.dot(state, state.T)
                xy = np.dot(state, observation.T)

                P = xx + state_covariances[k, t]

                observation_covariance += responsibilities[n, k] * (
                    yy - xy - xy.T + P
                )
    observation_covariance = observation_covariance / (N * T)
    return observation_covariance


def _update_transition_covariances(transition_matrix, state_means,
                                   state_covariances, pairwise_covariances):

    return filtermethods._em_transition_covariance(
        transition_matrices=transition_matrix,
        transition_offsets=np.array([0]),
        smoothed_state_means=state_means,
        smoothed_state_covariances=state_covariances,
        pairwise_covariances=pairwise_covariances
    )


def _update_component_weights(responsibilities):
    component_weights = responsibilities.sum(axis=0) / responsibilities.sum()
    return component_weights


def _update_initial_state_means(state_means):
    # initial state distributions
    return state_means[:, 0]


def _update_initial_state_covariance(initial_state_mean, state_means,
                                     state_covariances):

    initial_state_covariance = filtermethods._em_initial_state_covariance(
        initial_state_mean=initial_state_mean,
        smoothed_state_means=state_means,
        smoothed_state_covariances=state_covariances
    )

    return initial_state_covariance


def _expected_normal_logpdf(x=None, mean=None, covariance=None, precision=None,
                            mean_covariance=None):

    if precision is None:
        precision = np.linalg.pinv(covariance)

    expected_normal = 0
    expected_normal += \
        multivariate_normal.logpdf(x=x, mean=mean, cov=covariance)
    expected_normal += -0.5 * np.trace(np.dot(
            np.linalg.pinv(covariance),
            mean_covariance
        ))
    return expected_normal
