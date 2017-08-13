# cython: profile=True
# import cython
import numpy as np
# cimport numpy as np
from scipy import linalg
import mycanalysis.src.utils as utils


# @cython.boundscheck(False) # turn off bounds-checking for entire function
def _filter_predict(transition_matrix, transition_covariance,
                    current_state_mean, current_state_covariance):
    """
    Calculate the mean and covariance of :math:`P(x_{t+1} | z_{0:t})`
    Using the mean and covariance of :math:`P(x_t | z_{0:t})`, calculate the
    mean and covariance of :math:`P(x_{t+1} | z_{0:t})`.
    Parameters
    ----------
    transition_matrix : [n_dim_state, n_dim_state]
    transition_covariance : [n_dim_state, n_dim_state]
    current_state_mean: [n_dim_state]
    current_state_covariance: [n_dim_state, n_dim_state]
    Returns
    -------
    predicted_state_mean : [n_dim_state] array
    predicted_state_covariance : [n_dim_state, n_dim_state]
    """
    predicted_state_mean = (
        np.dot(transition_matrix, current_state_mean)
    )
    predicted_state_covariance = (
        np.dot(transition_matrix,
               np.dot(current_state_covariance,
                      transition_matrix.T))
        + transition_covariance
    )

    return (predicted_state_mean, predicted_state_covariance)


# @cython.boundscheck(False) # turn off bounds-checking for entire function
def _filter_correct(observation_matrix, observation_precision,
                    predicted_state_mean, predicted_state_covariance,
                    observation):
    """
    Correct a predicted state with a Kalman Filter update
    Incorporate observation `observation` from time `t` to turn
    :math:`P(x_t | z_{0:t-1})` into :math:`P(x_t | z_{0:t})`
    Parameters
    ----------
    observation_matrix : [n_dim_obs, n_dim_state]
    observation_covariance : [n_dim_obs, n_dim_obs]
    predicted_state_mean : [n_dim_state]
    predicted_state_covariance : [n_dim_state, n_dim_state]
    observation : [n_dim_obs]
    Returns
    -------
    kalman_gain : [n_dim_state, n_dim_obs]
    corrected_state_mean : [n_dim_state]
    corrected_state_covariance : [n_dim_state, n_dim_state]
    """
    n_dim_obs, n_dim_state = observation_matrix.shape

    predicted_observation_mean = np.dot(
        observation_matrix, predicted_state_mean
    )

    # usd woodbury identity
    predicted_state_precision = linalg.pinv(
        predicted_state_covariance
    )

    predicted_observation_precision = utils.woodbury_inversion(
        Ainv=observation_precision,
        U=observation_matrix,
        Cinv=predicted_state_precision,
        V=observation_matrix.T
    )

    kalman_gain = np.dot(
        predicted_state_covariance,
        np.dot(
            observation_matrix.T,
            predicted_observation_precision
        )
    )

    corrected_state_mean = (
        predicted_state_mean
        + np.dot(kalman_gain, observation - predicted_observation_mean)
    )
    corrected_state_covariance = (
        predicted_state_covariance
        - np.dot(kalman_gain,
                 np.dot(observation_matrix,
                        predicted_state_covariance))
    )

    return (kalman_gain, corrected_state_mean,
            corrected_state_covariance)


# @cython.boundscheck(False) # turn off bounds-checking for entire function
def _filter(transition_matrix, observation_matrix, transition_covariance,
            observation_precision, initial_state_mean,
            initial_state_covariance, observations):
    """
    Apply the Kalman Filter
    Calculate posterior distribution over hidden states given observations up
    to and including the current time step.
    Parameters
    ----------
    transition_matrix : [n_dim_state,n_dim_state]
    observation_matrix : [n_dim_obs, n_dim_state]
    transition_covariance : [n_dim_state,n_dim_state]
    observation_precision : [n_dim_obs, n_dim_obs]
    initial_state_mean : [n_dim_state]
    initial_state_covariance : [n_dim_state, n_dim_state]
    observations : [n_timesteps, n_dim_obs]
    Returns
    -------
    predicted_state_means : [n_timesteps, n_dim_state]
    predicted_state_covariances : [n_timesteps, n_dim_state, n_dim_state]
    kalman_gains : [n_timesteps, n_dim_state]
    filtered_state_means : [n_timesteps, n_dim_state]
    filtered_state_covariances : [n_timesteps, n_dim_state]
    """
    n_timesteps = observations.shape[0]
    n_dim_obs = observations.shape[1]
    n_dim_state = initial_state_mean.shape[0]

    predicted_state_means = np.zeros((n_timesteps, n_dim_state))
    predicted_state_covariances = np.zeros(
        (n_timesteps, n_dim_state, n_dim_state)
    )
    kalman_gains = np.zeros((n_timesteps, n_dim_state, n_dim_obs))
    filtered_state_means = np.zeros((n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros(
        (n_timesteps, n_dim_state, n_dim_state)
    )

    for t in range(n_timesteps):
        if t == 0:
            predicted_state_means[t] = initial_state_mean
            predicted_state_covariances[t] = initial_state_covariance
        else:
            predicted_state_means[t], predicted_state_covariances[t] = (
                _filter_predict(
                    transition_matrix,
                    transition_covariance,
                    filtered_state_means[t - 1],
                    filtered_state_covariances[t - 1]
                )
            )

        (kalman_gains[t], filtered_state_means[t],
         filtered_state_covariances[t]) = (
            _filter_correct(observation_matrix,
                            observation_precision,
                            predicted_state_means[t],
                            predicted_state_covariances[t],
                            observations[t]
                            )
                    )

    return (predicted_state_means, predicted_state_covariances,
            kalman_gains, filtered_state_means,
            filtered_state_covariances)


# @cython.boundscheck(False) # turn off bounds-checking for entire function
def _smooth_update(transition_matrix, filtered_state_mean,
                   filtered_state_covariance, predicted_state_mean,
                   predicted_state_covariance, next_smoothed_state_mean,
                   next_smoothed_state_covariance):
    """
    Correct a predicted state with a Kalman Smoother update
    Calculates posterior distribution of the hidden state at time `t` given the
    observations all observations via Kalman Smoothing.
    Parameters
    ----------
    transition_matrix : [n_dim_state, n_dim_state]
    filtered_state_mean : [n_dim_state]
    filtered_state_covariance : [n_dim_state, n_dim_state]
    predicted_state_mean : [n_dim_state]
    predicted_state_covariance : [n_dim_state, n_dim_state]
    next_smoothed_state_mean : [n_dim_state]
    next_smoothed_state_covariance : [n_dim_state, n_dim_state]
    Returns
    -------
    smoothed_state_mean : [n_dim_state]
    smoothed_state_covariance : [n_dim_state, n_dim_state]
    kalman_smoothing_gain : [n_dim_state, n_dim_state]
    """

    kalman_smoothing_gain = (
        np.dot(filtered_state_covariance,
               np.dot(transition_matrix.T,
                      linalg.pinv(predicted_state_covariance)))
    )

    smoothed_state_mean = (
        filtered_state_mean
        + np.dot(kalman_smoothing_gain,
                 next_smoothed_state_mean - predicted_state_mean)
    )
    smoothed_state_covariance = (
        filtered_state_covariance
        + np.dot(kalman_smoothing_gain,
                 np.dot(
                    (next_smoothed_state_covariance
                        - predicted_state_covariance),
                    kalman_smoothing_gain.T
                 ))
    )

    return (smoothed_state_mean, smoothed_state_covariance,
            kalman_smoothing_gain)


# @cython.boundscheck(False) # turn off bounds-checking for entire function
def _smooth(transition_matrix, filtered_state_means,
            filtered_state_covariances, predicted_state_means,
            predicted_state_covariances):
    """
    Apply the Kalman Smoother
    Estimate the hidden state at time for each time step given all
    observations.
    Parameters
    ----------
    transition_matrix : [n_dim_state, n_dim_state]
    filtered_state_means : [n_timesteps, n_dim_state]
    filtered_state_covariances : [n_timesteps, n_dim_state, n_dim_state]
    predicted_state_means : [n_timesteps, n_dim_state]
    predicted_state_covariances : [n_timesteps, n_dim_state, n_dim_state]
    Returns
    -------
    smoothed_state_means : [n_timesteps, n_dim_state]
    smoothed_state_covariances : [n_timesteps, n_dim_state, n_dim_state]
    kalman_smoothing_gains : [n_timesteps-1, n_dim_state, n_dim_state]
    """
    n_timesteps, n_dim_state = filtered_state_means.shape

    smoothed_state_means = np.zeros((n_timesteps, n_dim_state))
    smoothed_state_covariances = np.zeros((n_timesteps, n_dim_state,
                                           n_dim_state))
    kalman_smoothing_gains = np.zeros((n_timesteps - 1, n_dim_state,
                                       n_dim_state))

    smoothed_state_means[-1] = filtered_state_means[-1]
    smoothed_state_covariances[-1] = filtered_state_covariances[-1]

    for t in reversed(range(n_timesteps - 1)):
        (smoothed_state_means[t], smoothed_state_covariances[t],
         kalman_smoothing_gains[t]) = (
            _smooth_update(
                transition_matrix,
                filtered_state_means[t],
                filtered_state_covariances[t],
                predicted_state_means[t + 1],
                predicted_state_covariances[t + 1],
                smoothed_state_means[t + 1],
                smoothed_state_covariances[t + 1]
            )
        )
    return (smoothed_state_means, smoothed_state_covariances,
            kalman_smoothing_gains)


# @cython.boundscheck(False) # turn off bounds-checking for entire function
def _smooth_pair(smoothed_state_covariances, kalman_smoothing_gains):
    """
    Calculate pairwise covariance between hidden states
    Calculate covariance between hidden states at :math:`t` and :math:`t-1` for
    all time step pairs
    Parameters
    ----------
    smoothed_state_covariances : [n_timesteps, n_dim_state, n_dim_state]
    kalman_smoothing_gains : [n_timesteps-1, n_dim_state, n_dim_state]
    Returns
    -------
    pairwise_covariances : [n_timesteps, n_dim_state, n_dim_state]
    """
    n_timesteps, n_dim_state, _ = smoothed_state_covariances.shape
    pairwise_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    for t in range(1, n_timesteps):
        pairwise_covariances[t] = (
            np.dot(smoothed_state_covariances[t],
                   kalman_smoothing_gains[t - 1].T)
        )
    return pairwise_covariances


def test(N=10000, T=20):
    data = np.random.random((N, T))

    transition_matrix = np.ones([1, 1])
    transition_covariance = np.ones([1, 1])
    observation_matrix = np.ones([N, 1])
    observation_covariance = np.ones([1, 1])
    observation_precision = linalg.pinv(observation_covariance)
    responsibilities = np.ones(N)

    initial_state_mean = np.zeros(1)
    initial_state_covariance = np.ones([1, 1])

    block_observation_precision = linalg.block_diag(
        *[observation_precision * r for r in responsibilities]
    )

    (predicted_state_means, predicted_state_covariances,
     kalman_gains, filtered_state_means, filtered_state_covariances) = (
        _filter(
            transition_matrix, observation_matrix,
            transition_covariance, block_observation_precision,
            initial_state_mean, initial_state_covariance, data.T
        )
    )

    (smoothed_state_means, smoothed_state_covariances,
        kalman_smoothing_gains) = (
        _smooth(
            transition_matrix, filtered_state_means,
            filtered_state_covariances, predicted_state_means,
            predicted_state_covariances
        )
    )
