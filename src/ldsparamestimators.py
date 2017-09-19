import numpy as np


def update_observation_matrix():
    pass


def update_observation_covariance(observations, observation_matrix,
                                  state_means, state_covariances,
                                  weights=None):
    """
    Estimate observation covariance
    from weighted observations and a state estimate

    observations [n_obs, t_timepoints, n_dim_obs] matrix of observations
    weights [n_obs] array of weights for each observation
    observation_matrix[n_dim_obs, n_dim_state]
        projects state into observation space

    state_means [t_timepoints, n_dim_state]
    state_covariances [t_timepoints, n_dim_state, n_dim_state]
    """
    if weights is None:
        weights = np.ones(observations.shape[0])

    included = np.where(np.logical_not(np.isclose(weights, 0)))
    observations = observations[included]
    weights = weights[included]

    total_weight = weights.sum()

    n_obs = observations.shape[0]
    n_dim_obs, n_dim_state = observation_matrix.shape[:2]
    t_timepoints = state_means.shape[0]

    observation_covariance = np.zeros((n_dim_obs, n_dim_obs))

    x_x = np.zeros((n_dim_state, n_dim_state))  # raw second moment
    for t in range(t_timepoints):
        x_x += np.outer(state_means[t], state_means[t]) + state_covariances[t]

    for n in range(n_obs):
        observation = observations[n]
        y_y = np.zeros((n_dim_obs, n_dim_obs))
        x_y = np.zeros((n_dim_state, n_dim_obs))
        for t in range(t_timepoints):
            y_y += np.outer(observation[t], observation[t])
            x_y += np.outer(state_means[t], observation[t])

        observation_covariance += \
            weights[n] * (y_y - (2 * np.dot(observation_matrix, x_y)))

    observation_covariance /= (total_weight * t_timepoints)

    observation_covariance += \
        np.dot(observation_matrix, np.dot(x_x, observation_matrix.T)) \
        / t_timepoints

    return observation_covariance


def update_transition_matrix():
    pass


def update_transition_covariance(state_means, state_covariances,
                                 pairwise_covariances, transition_matrix):

    """
    state_covariances [t_timepoints, n_dim_state, n_dim_state]
    pairwise_covariances [t_timepoints, n_dim state, n_dim_state]
        the first entry doesnt mean anything
    transition_matrix [n_dim_state, n_dim_state]
        projects state forward in time
    """

    t_timepoints, n_dim_state = state_means.shape[:2]

    # raw second moment
    x_x = np.zeros((t_timepoints, n_dim_state, n_dim_state))

    # Cov(x1, x2) + E[x1]E[x2]
    x2_x1 = np.zeros((t_timepoints, n_dim_state, n_dim_state))

    for t in range(t_timepoints):
        x_x[t] = \
            np.outer(state_means[t], state_means[t]) + state_covariances[t]

    for t in range(1, t_timepoints):
        x2_x1[t] = np.outer(state_means[t], state_means[t-1]) + \
            pairwise_covariances[t]

    x2x1A = np.dot(x2_x1[1:].sum(axis=0), transition_matrix.T)

    AxxA = np.dot(transition_matrix,
                  np.dot(x_x[:-1].sum(axis=0), transition_matrix.T))

    transition_covariance = (x_x[1:].sum(axis=0) - x2x1A - x2x1A.T + AxxA) \
        / (t_timepoints - 1)

    return transition_covariance


def update_initial_state_mean(state_means):
    return state_means[0]


def update_initial_state_covariance(state_means, initial_state_mean,
                                    state_covariances):
    initial_state_covariance = (
        np.outer(state_means[0], state_means[0]) + state_covariances[0]
        - np.outer(state_means[0], initial_state_mean)
        - np.outer(initial_state_mean, state_means[0])
        + np.outer(initial_state_mean, initial_state_mean)
    )

    return initial_state_covariance
