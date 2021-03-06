# cython: profile=True
import cython
import numpy as np
cimport numpy as np
import scipy as sp
import functools
from multiprocessing import Pool
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal

import mycanalysis.src.ldsinference as filtermethods

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

def _estimate_states(
    data,
    transition_matrix,
    observation_matrix,
    transition_covariances,
    observation_covariance,
    initial_state_means,
    initial_state_covariances,
    responsibilities, 
    int K, int T, int U, int V, int processes):
    """
    estimates state sequences for all trajectories/clusters
    """

    cdef np.ndarray[dtype=np.float64_t, ndim=3] state_means = \
        np.zeros((K, T, U), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=4] state_covariances = \
        np.zeros((K, T, V, V), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=4] pairwise_covariances = \
        np.zeros((K, T, V, V), dtype=np.float64)

    cdef int k
    for k in range(K):
        results = _single_estimate_states(
            data,
            transition_matrix,
            observation_matrix,
            transition_covariances[k],
            observation_covariance,
            initial_state_means[k],
            initial_state_covariances[k],
            responsibilities[:, k]
        )
        
        if results[0] is not None:
            state_means[k] = results[0]
            state_covariances[k] = results[1]
            pairwise_covariances[k] = results[2]

    return (np.asarray(state_means),
        np.asarray(state_covariances), np.asarray(pairwise_covariances))


def _single_estimate_states(data, transition_matrix, observation_matrix,
                            transition_covariance, observation_covariance,
                            initial_state_mean, initial_state_covariance,
                            responsibilities):
    """
    estimate states for a single component
    """
    f, included = _initialize_filter(
        transition_matrix=transition_matrix,
        observation_matrix=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        responsibilities=responsibilities
    )

    means, covariances, pairwise_covariances = \
        _filter_and_smooth(f, included, data)

    return means, covariances, pairwise_covariances


def _estimate_responsibilities(observation, state_means, state_covariances,
                               observation_covariance, component_weights):
    """
    estimate responsibilities for an observation
    """
    expected_conditional_likelihoods = \
        _expected_conditional_likelihoods(
            observation,
            state_means=state_means,
            state_covariances=state_covariances,
            observation_covariance=observation_covariance,
            component_weights=component_weights
        )

    responsibilities = np.exp(
            expected_conditional_likelihoods -
            logsumexp(expected_conditional_likelihoods)
    )

    return responsibilities


################
#  likelihoods #
################


cdef _expected_observation_likelihood(observation, responsibilities,
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


cdef _expected_conditional_likelihoods(observation, state_means,
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
                x=np.array(observation[t]).reshape(-1),
                mean=np.array(state_means[k, t]).reshape(-1),
                covariance=observation_covariance,
                precision=observation_precision,
                mean_covariance=state_covariances[k, t]
            )

        conditional_expected_likelihoods[k] = \
            np.log(component_weights[k]) + sub

    return conditional_expected_likelihoods


cdef _expected_sequence_likelihood(initial_state_mean, initial_state_covariance,
                                  transition_covariance, state_means,
                                  state_covariances, pairwise_covariances):

    """
    expectation of state sequence likelihood
    expectation taken over estimated state distribution
    """
    initial_state_precision = np.linalg.pinv(initial_state_covariance)
    expected_sequence_likelihood = 0
    expected_sequence_likelihood += _expected_normal_logpdf(
        x=state_means[0],
        mean=initial_state_mean,
        covariance=initial_state_covariance,
        precision=initial_state_precision,
        mean_covariance=state_covariances[0]
    )

    T = state_means.shape[0]
    for t in range(1, T):
        covariance = state_covariances[t] + state_covariances[t-1] - \
            (2 * pairwise_covariances[t])
        precision = np.linalg.pinv(covariance)

        expected_sequence_likelihood += _expected_normal_logpdf(
            x=state_means[t],
            mean=state_means[t-1],
            covariance=transition_covariance,
            precision=precision,
            mean_covariance=covariance
        )

    return expected_sequence_likelihood


#############
# entropies #
#############

cdef _assignment_entropy(responsibilities):
    """
    entropy of posterior assignment estimate
    """
    active_responsibilities = responsibilities[
        np.logical_not(np.isclose(responsibilities, 0))
        ]
    assignment_entropy = -1 * \
        (active_responsibilities * np.log(active_responsibilities)).sum()
    return assignment_entropy


cdef _state_sequence_entropy(state_covariances, pairwise_covariances):
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

    entropy = 0
    expected_likelihood = 0

    n_observations = data.shape[0]
    n_components = len(component_weights)

    for n in range(n_observations):
        expected_likelihood += _expected_observation_likelihood(
            observation=data[n],
            responsibilities=responsibilities[n],
            state_means=state_means,
            state_covariances=state_covariances,
            observation_covariance=observation_covariance,
            component_weights=component_weights
        )

        entropy += _assignment_entropy(responsibilities=responsibilities[n])


    for k in range(n_components):
        expected_likelihood += _expected_sequence_likelihood(
            initial_state_mean=np.array(initial_state_means[k]).reshape(-1),
            initial_state_covariance=initial_state_covariances[k],
            transition_covariance=transition_covariances[k],
            state_means=state_means[k],
            state_covariances=state_covariances[k],
            pairwise_covariances=pairwise_covariances[k]
        )
        entropy += _state_sequence_entropy(
            state_covariances=state_covariances[k],
            pairwise_covariances=pairwise_covariances[k]
        )

    elbo = expected_likelihood + entropy

    return elbo, expected_likelihood, entropy


################
# helper methods
################

cdef _initialize_filter(transition_matrix, observation_matrix,
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

        observation_precision = sp.linalg.pinv(observation_covariance)
        block_observation_precision = sp.linalg.block_diag(
            *[observation_precision * r for r in responsibilities[included]]
        )

        block_observation_matrix = np.tile(
            observation_matrix, (observation_dim, 1)
        )

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


cdef float _expected_normal_logpdf(x, mean, covariance, precision, mean_covariance):

    cdef float expected_normal = 0
    expected_normal += \
        normal_logpdf(x=x, mean=mean, cov=covariance)

    expected_normal += -0.5 * np.trace(np.dot(
            precision,
            mean_covariance
        ))
    return expected_normal

cdef float normal_logpdf(
    np.ndarray[np.float64_t, ndim=1] x,
    np.ndarray[np.float64_t, ndim=1] mean,
    np.ndarray[np.float64_t, ndim=2] cov):
    
    cdef np.ndarray[np.float64_t, ndim=1] residual
    residual = mean - x
    cdef float quad
    quad = np.dot(
        residual, np.dot(np.linalg.inv(cov), residual.T)
    )
    
    cdef float const = np.log(np.linalg.det(2 * np.pi * cov))
    
    return -0.5 * (quad + const)


cdef _filter_and_smooth(f, included, data):
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
         kalman_gains, filtered_state_means, filtered_state_covariances) = \
            filtermethods._filter(
                transition_matrix=f['transition_matrix'],
                observation_matrix=f['observation_matrix'],
                transition_covariance=f['transition_covariance'],
                observation_precision=f['observation_precision'],
                initial_state_mean=f['initial_state_mean'],
                initial_state_covariance=f['initial_state_covariance'],
                observations=Z
            )

        (smoothed_state_means, smoothed_state_covariances,
            kalman_smoothing_gains) = (
            filtermethods._smooth(
                transition_matrix=f['transition_matrix'],
                filtered_state_means=filtered_state_means,
                filtered_state_covariances=filtered_state_covariances,
                predicted_state_means=predicted_state_means,
                predicted_state_covariances=predicted_state_covariances
            )
        )

        pairwise_covariances = filtermethods._smooth_pair(
            smoothed_state_covariances=smoothed_state_covariances,
            kalman_smoothing_gains=kalman_smoothing_gains
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
