import numpy as np
import scipy as sp
import ldsinference
import ldsparamestimators as params

from mycanalysis.src.utils import (expected_normal_logpdf, normal_logpdf,
                                   quadratic_expectation, quadratic, _expected_log_likelihoods)
from collections import defaultdict
from scipy.stats import multivariate_normal


class LinearDynamicalSystem:
    def __init__(self, n_dim_state=1, n_dim_obs=1, n_obs=1, weights=None,
                 initial_state_mean=None, initial_state_covariance=None,
                 transition_matrix=None, transition_covariance=None,
                 observation_matrix=None, observation_covariance=None,
                 fixed_params=[]):

        self.n_dim_state = n_dim_state  # dimension of state space
        self.n_dim_obs = n_dim_obs      # dimension of single observation

        self.n_obs = n_obs              # number of obs (ie multiple sensors)

        # weights reflecting confidence in each observation f
        if weights is None:
            self.weights = np.ones(n_obs)
        else:
            self.weights = weights

        # model parameters
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.transition_matrix = transition_matrix
        self.transition_covariance = transition_covariance

        # make block observation matrix and observation covariance
        # this is for the case where we are taking multiple observations
        # at a timepoint
        self.observation_matrix = observation_matrix
        self.observation_covariance = observation_covariance
        self.observation_precision = \
            np.linalg.pinv(self.observation_covariance)

        if n_obs == 1:
            self.logdet_observation_covariance = \
                np.linalg.slogdet(self.observation_covariance)[1]

        else:
            self.observation_matrix = \
                make_block_observation_matrix(observation_matrix, n_obs)

            self.observation_covariance, self.observation_precision = \
                make_block_observation_covariance(
                    observation_covariance,
                    np.linalg.pinv(observation_covariance),
                    self.weights
                )
            self.logdet_observation_covariance = \
                np.linalg.slogdet(observation_covariance)[1] * n_obs \
                + (np.log(weights) ** n_dim_obs).sum()

        # will hold state distributions
        self.state_means = None
        self.state_covariances = None
        self.pariwise_covariances = None

        # cache these so se dont need to keep recomputing them
        self.projected_state_means = None
        self.projected_state_covariances = None

        self.fixed = defaultdict(bool)
        for param in fixed_params:
            self.fixed[param] = True

    def update_parameters(self, data=None):
        """
        update parameters specified for update at initialization
        using data and current state estimates stored in model
        """
        # update observation matrix and observation covariance
        if not self.fixed['observation_matrix']:
            observation_matrix = self.update_observation_matrix(inplace=False)
        else:
            observation_matrix = self.observation_matrix

        if not self.fixed['observation_covariance']:
            observation_covariance = \
                self.update_observation_covariance(data, inplace=False)
        else:
            observation_covariance = self.observation_covariance

        self.observation_matrix = observation_matrix
        self.observation_covariance = observation_covariance

        # update transition matrix and transition covariance
        if not self.fixed['transition_matrix']:
            transition_matrix = self.update_transition_matrix(inplace=False)
        else:
            transition_matrix = self.transition_matrix

        if not self.fixed['transition_covariance']:
            transition_covariance = \
                self.update_transition_covariance(inplace=False)
        else:
            transition_covariance = self.transition_covariance

        self.transition_matrix = transition_matrix
        self.transition_covariance = transition_covariance

        # update initial state distribution
        if not self.fixed['initial_state_mean']:
            initial_state_mean = self.update_initial_state_mean(inplace=False)
        else:
            initial_state_mean = self.initial_state_mean

        if not self.fixed['initial_state_covariance']:
            initial_state_covariance = \
                self.update_initial_state_covariance(inplace=False)
        else:
            initial_state_covariance = self.initial_state_covariance

        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance

        return (observation_matrix, observation_covariance,
                transition_matrix, transition_covariance,
                initial_state_mean, initial_state_covariance)

    def update_observation_covariance(self, data, weights=None, inplace=True):
        observation_covariance = params.update_observation_covariance(
            observations=data,
            observation_matrix=self.observation_matrix,
            state_means=self.state_means,
            state_covariances=self.state_covariances,
            weights=weights
        )

        if inplace:
            self.observation_covariance = observation_covariance
            self.logdet_observation_covariance = \
                np.linalg.slogdet(observation_covariance)[1]

        return observation_covariance

    def update_transition_covariance(self, inplace=True):
        transition_covariance = params.update_transition_covariance(
            state_means=self.state_means,
            state_covariances=self.state_covariances,
            pairwise_covariances=self.pairwise_covariances,
            transition_matrix=self.transition_matrix
        )

        if inplace:
            self.transition_covariance = transition_covariance

        return transition_covariance

    def update_initial_state_mean(self, inplace=True):
        initial_state_mean = self.state_means[0][:, np.newaxis]

        if inplace:
            self.initial_state_mean = initial_state_mean

        return initial_state_mean

    def update_initial_state_covariance(self, inplace=True):
        initial_state_covariance = params.update_initial_state_covariance(
            state_means=self.state_means,
            initial_state_mean=self.initial_state_mean,
            state_covariances=self.state_covariances
        )

        if inplace:
            self.initial_state_covariance = initial_state_covariance

        return initial_state_covariance

    def filter(self, data, weights):
        (predicted_state_means, predicted_state_covariances,
         kalman_gains, filtered_state_means, filtered_state_covariances) = \
            ldsinference._filter(
                transition_matrix=self.transition_matrix,
                observation_matrix=self.observation_matrix,
                transition_covariance=self.transition_covariance,
                observation_precision=self.observation_precision,
                initial_state_mean=self.initial_state_mean,
                initial_state_covariance=self.initial_state_covariance,
                observations=data)

    def smooth(self, data, cache_projected=True):
        (predicted_state_means, predicted_state_covariances,
         kalman_gains, filtered_state_means, filtered_state_covariances) = \
            ldsinference._filter(
                transition_matrix=self.transition_matrix,
                observation_matrix=self.observation_matrix,
                transition_covariance=self.transition_covariance,
                observation_precision=self.observation_precision,
                initial_state_mean=self.initial_state_mean,
                initial_state_covariance=self.initial_state_covariance,
                observations=data)

        (smoothed_state_means, smoothed_state_covariances,
         kalman_smoothing_gains) = \
            ldsinference._smooth(
                transition_matrix=self.transition_matrix,
                filtered_state_means=filtered_state_means,
                filtered_state_covariances=filtered_state_covariances,
                predicted_state_means=predicted_state_means,
                predicted_state_covariances=predicted_state_covariances
            )

        pairwise_covariances = ldsinference._smooth_pair(
            smoothed_state_covariances,
            kalman_smoothing_gains
        )

        self.state_means = smoothed_state_means
        self.state_covariances = smoothed_state_covariances
        self.pairwise_covariances = pairwise_covariances

        if cache_projected:
            self.project_states()

    def smooth_multiple(self, data, weights=None, cache_projected=True):
        """
        data [n_obs, t_timepoints, n_dim obs]
        jointly estimates states for multiple concurrent observations
        """

        n_obs, t_timepoints = data.shape[:2]

        if weights is None:
            weights = np.ones(n_obs)

        included = np.where(np.logical_not(np.isclose(weights, 0)))[0]

        X = data[included].view().swapaxes(0, 1)
        X.shape = (t_timepoints, -1)

        # make a filter for concatenated observations
        fused_filter = LinearDynamicalSystem(
            n_dim_state=self.n_dim_state,
            n_dim_obs=self.n_dim_obs,
            n_obs=included.size,
            weights=weights[included],
            initial_state_mean=self.initial_state_mean,
            initial_state_covariance=self.initial_state_covariance,
            transition_matrix=self.transition_matrix,
            transition_covariance=self.transition_covariance,
            observation_matrix=self.observation_matrix,
            observation_covariance=self.observation_covariance
        )

        # estimate states on fused filter
        fused_filter.smooth(X, cache_projected=False)

        # update state estimates
        self.state_means = fused_filter.state_means
        self.state_covariances = fused_filter.state_covariances
        self.pairwise_covariances = fused_filter.pairwise_covariances

        if cache_projected:
            self.project_states()

    def information_smooth(self, data, weights, cache_projected=True):
        (predicted_state_means, predicted_state_covariances,
         predicted_information, filtered_state_means,
         filtered_state_covariances, filtered_information) = \
            ldsinference._information_filter(
                transition_matrix=self.transition_matrix,
                transition_covariance=self.transition_covariance,
                observation_matrix=self.observation_matrix,
                observation_precision=self.observation_precision,
                initial_state_mean=self.initial_state_mean,
                initial_state_precision=np.linalg.pinv(
                    self.initial_state_covariance),
                observations=data,
                weights=weights
            )

        (smoothed_state_means, smoothed_state_covariances,
         kalman_smoothing_gains) = \
            ldsinference._smooth(
                transition_matrix=self.transition_matrix,
                filtered_state_means=filtered_state_means,
                filtered_state_covariances=filtered_state_covariances,
                predicted_state_means=predicted_state_means,
                predicted_state_covariances=predicted_state_covariances
            )

        pairwise_covariances = ldsinference._smooth_pair(
            smoothed_state_covariances,
            kalman_smoothing_gains
        )

        self.state_means = smoothed_state_means
        self.state_covariances = smoothed_state_covariances
        self.pairwise_covariances = pairwise_covariances

        if cache_projected:
            self.project_states()

    def information_smooth_multiple(self, data, weights=None,
                                    cache_projected=True):
        """
        data [n_obs, t_timepoints, n_dim obs]
        jointly estimates states for multiple concurrent observations
        """

        n_obs, t_timepoints = data.shape[:2]

        if weights is None:
            weights = np.ones(n_obs)

        included = np.where(np.logical_not(np.isclose(weights, 0)))[0]

        X = data[included].view().swapaxes(0, 1)
        X.shape = (t_timepoints, -1)

        # make a filter for concatenated observations

        # estimate states on fused filter
        self.information_smooth(X, weights[included], cache_projected=True)

        if cache_projected:
            self.project_states()

    def log_likelihoods(self, data):
        """
        data [n_obs, t_timepoints, n_dim_obs] or [t_timepoints, n_dim_obs]

        returns log_likelihoods
            [n_obs, t_timepoints] or [t_timepoints]
        """

        if data.ndim == 3:
            n_obs, t_timepoints = data.shape[:2]
        if data.ndim == 2:
            t_timepoints = data.shape[0]
            n_obs = 1
            data = data[np.newaxis]

        log_likelihoods = np.zeros((n_obs, t_timepoints))
        for n in range(n_obs):
            for t in range(t_timepoints):
                mean = np.dot(self.observation_matrix, self.state_means[t])
                covariance = np.dot(
                    np.dot(self.observation_matrix, self.state_covariances[t]),
                    self.observation_matrix.T
                ) + self.observation_covariance

                precision = np.linalg.pinv(covariance)
                log_likelihoods[n, t] = normal_logpdf(
                    x=data[n, t],
                    mean=mean,
                    covariance=self.observation_covariance,
                    precision=precision,
                )

        return log_likelihoods.squeeze()

    def expected_log_likelihoods(self, data):
        """
        data [n_obs, t_timepoints, n_dim_obs] or [t_timepoints, n_dim_obs]

        returns expected_log_likelihoods
            [n_obs, t_timepoints] or [t_timepoints]
        """
        result = _expected_log_likelihoods(
            data, self.observation_precision, self.projected_state_means,
            self.projected_state_covariances, self.logdet_observation_covariance
        )

        return np.asarray(result)

    def expected_squared_error(self, data):
        """
        data [n_obs, t_timepoints, n_dim_obs] or [t_timepoints, n_dim_obs]

        returns expected_log_likelihoods
            [n_obs, t_timepoints] or [t_timepoints]
        """

        if data.ndim == 3:
            n_obs, t_timepoints = data.shape[:2]
        if data.ndim == 2:
            t_timepoints = data.shape[0]
            n_obs = 1
            data = data[np.newaxis]

        expected_squared_error = np.zeros((n_obs, t_timepoints))
        for n in range(n_obs):
            for t in range(t_timepoints):
                residual = data[n, t] - \
                    np.dot(self.observation_matrix, self.state_means[t])
                expected_squared_error[n, t] = quadratic_expectation(
                    x=residual,
                    A=self.observation_precision,
                    V=self.state_covariances[t]
                )
        return expected_squared_error.squeeze()

    def expected_sequence_likelihood(self):
        """
        expectation of state sequence likelihood
        expectation taken over estimated state distribution
        """
        initial_state_precision = np.linalg.inv(self.initial_state_covariance)
        expected_sequence_likelihood = 0
        expected_sequence_likelihood += expected_normal_logpdf(
            x=self.state_means[0],
            mean=self.initial_state_mean,
            covariance=self.initial_state_covariance,
            precision=initial_state_precision,
            V=self.state_covariances[0]
        )

        t_timepoints = self.state_means.shape[0]
        transition_precision = np.linalg.inv(self.transition_covariance)
        for t in range(1, t_timepoints):
            covariance = self.state_covariances[t] + \
                         self.state_covariances[t-1] - \
                         (2 * self.pairwise_covariances[t])

            expected_sequence_likelihood += expected_normal_logpdf(
                x=self.state_means[t],
                mean=self.state_means[t-1],
                covariance=self.transition_covariance,
                precision=transition_precision,
                V=covariance
            )

        return expected_sequence_likelihood

    def sequence_entropy(self):
        t_timepoints = self.state_covariances.shape[0]
        state_sequence_entropy = 0
        state_sequence_entropy += \
            multivariate_normal.entropy(cov=self.state_covariances[0])

        for t in range(1, t_timepoints):
            # covariance of state at time t given state at time t-1
            covariance = self.state_covariances[t] - np.linalg.multi_dot([
                self.pairwise_covariances[t],
                np.linalg.pinv(self.state_covariances[t-1]),
                self.pairwise_covariances[t]
            ])

            state_sequence_entropy += \
                multivariate_normal.entropy(cov=covariance)

        return state_sequence_entropy

    def project_states(self):
        t_timepoints = self.state_means.shape[0]
        dim = self.n_dim_obs * self.n_obs
        projected_state_means = np.zeros((t_timepoints, dim))
        projected_state_covariances = np.zeros((t_timepoints, dim, dim))

        for t in range(t_timepoints):
            projected_state_means[t] = \
                np.dot(self.observation_matrix, self.state_means[t])
            projected_state_covariances[t] = np.dot(
                self.observation_matrix,
                np.dot(self.state_covariances[t],
                       self.observation_matrix.T)
            )

        self.projected_state_means = projected_state_means
        self.projected_state_covariances = projected_state_covariances


def make_block_observation_matrix(observation_matrix, n_obs):
    block_observation_matrix = \
        np.tile(observation_matrix, (n_obs, 1))
    return block_observation_matrix


def make_block_observation_covariance(observation_covariance,
                                      observation_precision, weights):

    block_observation_covariance = sp.linalg.block_diag(
        *[observation_covariance / r for r in weights]
    )

    block_observation_precision = sp.linalg.block_diag(
        *[observation_precision * r for r in weights]
    )
    return block_observation_covariance, block_observation_precision
