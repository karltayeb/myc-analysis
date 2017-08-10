import pykalman.standard as filtermethods
import mycanalysis.src.ldsinference as ldsinference
import mycanalysis.src.utils as utils
import numpy as np
import scipy as sp
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal


####################
# parameter updates
####################

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

                observation_covariance += responsibilities[n, k] * (
                    yy - xy - xy.T + xx + state_covariances[k, t]
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


#####################
# variational updates
#####################

def _estimate_states(data, transition_matrix, observation_matrix,
                     transition_covariances, observation_covariance,
                     initial_state_means, initial_state_covariances,
                     responsibilities,
                     state_means, state_covariances, pairwise_covariances):
    """
    estimate states for a single component
    """
    k_components = transition_covariances.shape[0]
    for k in range(k_components):
        f, included = _initialize_filter(
            transition_matrix=transition_matrix,
            observation_matrix=observation_matrix,
            transition_covariance=transition_covariances[k],
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_means[k],
            initial_state_covariance=initial_state_covariances[k],
            responsibilities=responsibilities[:, k]
        )

        if f is not None:
            state_means[k], state_covariances[k], pairwise_covariances[k] = \
                _filter_and_smooth(f, included, data)


def _single_estimate_states(k, data, transition_matrix, observation_matrix,
                            transition_covariances, observation_covariance,
                            initial_state_means, initial_state_covariances,
                            responsibilities):
    """
    estimate states for a single component
    """
    f, included = _initialize_filter(
        transition_matrix=transition_matrix,
        observation_matrix=observation_matrix,
        transition_covariance=transition_covariances[k],
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_means[k],
        initial_state_covariance=initial_state_covariances[k],
        responsibilities=responsibilities[:, k]
    )

    means, covariances, pairwise_covariances = \
        _filter_and_smooth(f, included, data)

    return means, covariances, pairwise_covariances


def _estimate_responsibilities(observation, state_means, state_covariances,
                               observation_covariance, observation_precision,
                               component_weights):
    """
    estimate responsibilities for an observation

    arguments = {
        'state_means': state_means,
        'state_covariances': state_covariances,
        'observation_covariance': observation_covariance,
        'component_weights': component_weights,
    }
    """

    k_components = component_weights.shape[0]
    t_timepoints = observation.shape[0]

    expected_squared_errors = np.zeros(k_components)

    for k in range(k_components):
        for t in range(t_timepoints):
            expected_squared_errors[k] += -0.5 * \
                utils.quadratic_expectation(
                    x=(observation[t] - state_means[k, t]),
                    A=observation_precision,
                    V=state_covariances[k, t]
                )
    # expected_conditional_likelihoods = \
    #    _expected_conditional_likelihoods(observation, **arguments)

    responsibilities = np.exp(
            expected_squared_errors -
            logsumexp(expected_squared_errors)
    )

    return responsibilities


################
#  likelihoods #
################


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

    if observation.ndim == 1:
        observation = np.expand_dims(observation, 1)

    T = observation.shape[0]
    K = component_weights.shape[0]

    conditional_expected_likelihoods = np.zeros(K)

    observation_precision = np.linalg.pinv(observation_covariance)
    for k in range(K):
        sub = 0
        for t in range(T):
            sub += utils.expected_normal_logpdf(
                x=observation[t],
                mean=state_means[k, t],
                covariance=observation_covariance,
                precision=observation_precision,
                V=state_covariances[k, t]
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
    initial_state_precision = np.linalg.inv(initial_state_covariance)
    expected_sequence_likelihood += utils.expected_normal_logpdf(
        x=state_means[0],
        mean=initial_state_mean,
        covariance=initial_state_covariance,
        precision=initial_state_precision,
        V=state_covariances[0]
    )

    T = state_means.shape[0]
    transition_precision = np.linalg.pinv(transition_covariance)
    for t in range(1, T):
        covariance = state_covariances[t] + state_covariances[t-1] - \
            (2 * pairwise_covariances[t])

        expected_sequence_likelihood += utils.expected_normal_logpdf(
            x=state_means[t],
            mean=state_means[t-1],
            covariance=transition_covariance,
            precision=transition_precision,
            V=covariance
        )

    return expected_sequence_likelihood


#############
# entropies #
#############

def _assignment_entropy(responsibilities):
    """
    entropy of posterior assignment estimate
    """
    active_responsibilities = responsibilities[
        np.logical_not(np.isclose(responsibilities, 0))
        ]
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
                np.linalg.pinv(state_covariances[t-1]),
                pairwise_covariances[t]
            ])
        state_sequence_entropy += multivariate_normal.entropy(cov=covariance)

    return state_sequence_entropy


def _elbo(data, responsibilities, state_means, state_covariances,
          pairwise_covariances, component_weights, observation_covariance,
          initial_state_means, initial_state_covariances,
          transition_covariances, processes=1):
    """
    computed the evidence lower bound of the data
    returns float: evidence lower bound of data
    """
    if data.ndim == 2:
        data = np.expand_dims(data, 2)

    n_observations = data.shape[0]
    t_timepoints = data.shape[1]
    k_components = component_weights.shape[0]

    entropy = 0
    expected_likelihood = 0
    observation_precision = np.linalg.inv(observation_covariance)
    for n in range(n_observations):
        for k in range(k_components):
            sub = 0
            for t in range(t_timepoints):
                sub += utils.quadratic_expectation(
                    x=(data[n, t] - state_means[k, t]),
                    A=observation_precision,
                    V=state_covariances[k, t]
                )
            expected_likelihood += responsibilities[n, k] * (
                np.log(component_weights[k]) - 0.5 * sub
            )

    expected_likelihood += -0.5 * n_observations * t_timepoints * (
        np.log(np.linalg.det(2 * np.pi * observation_covariance))
    )

    for n in range(n_observations):
        entropy += _assignment_entropy(
            responsibilities[n]
        )

    for k in range(k_components):
        expected_likelihood += _expected_sequence_likelihood(
            initial_state_means[k],
            initial_state_covariances[k],
            transition_covariances[k],
            state_means[k],
            state_covariances[k],
            pairwise_covariances[k]
        )

        entropy += _state_sequence_entropy(
            state_covariances[k],
            pairwise_covariances[k]
        )

    elbo = expected_likelihood + entropy

    return elbo, expected_likelihood, entropy


################
# helper methods
################

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
    observation_precision = np.linalg.inv(observation_covariance)

    if observation_dim > 0:
        block_observation_covariance = sp.linalg.block_diag(
            *[observation_covariance / r for r in responsibilities[included]]
            )

        block_observation_precision = sp.linalg.block_diag(
            *[observation_precision * r for r in responsibilities[included]]
            )

        block_observation_matrix = \
            np.tile(observation_matrix, (observation_dim, 1))

        f = {
            'transition_matrix': transition_matrix,
            'observation_matrix': block_observation_matrix,
            'transition_covariance': transition_covariance,
            'observation_covariance': block_observation_covariance,
            'observation_precision': block_observation_precision,
            'initial_state_mean': initial_state_mean,
            'initial_state_covariance': initial_state_covariance,
            'n_dim_state': 1, 'n_dim_obs': observation_dim
        }
    else:
        f = None

    return f, included


def _filter_and_smooth(f, included, data):
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
        Z = data[included].T

        (predicted_state_means, predicted_state_covariances,
         kalman_gains, filtered_state_means, filtered_state_covariances) = (
            ldsinference._filter(
                transition_matrix=f['transition_matrix'],
                observation_matrix=f['observation_matrix'],
                transition_covariance=f['transition_covariance'],
                observation_precision=f['observation_precision'],
                initial_state_mean=f['initial_state_mean'],
                initial_state_covariance=f['initial_state_covariance'],
                observations=Z
            )
        )

        (smoothed_state_means, smoothed_state_covariances,
            kalman_smoothing_gains) = (
            ldsinference._smooth(
                transition_matrix=f['transition_matrix'],
                filtered_state_means=filtered_state_means,
                filtered_state_covariances=filtered_state_covariances,
                predicted_state_means=predicted_state_means,
                predicted_state_covariances=predicted_state_covariances
            )
        )

        pairwise_covariances = ldsinference._smooth_pair(
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
