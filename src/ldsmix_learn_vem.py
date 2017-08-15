import sys
import os
import pickle
import numpy as np
import copy
import LDSMixture
from sklearn.datasets import make_spd_matrix


def random_model(N, K, T, U, V):
    responsibilities = np.random.uniform(1, 10, (N, K))
    responsibilities = \
        responsibilities / responsibilities.sum(axis=1)[:, np.newaxis]

    component_weights = np.random.uniform(1, 10, K)
    component_weights = component_weights / component_weights.sum()
    initialization = {
        'K': K, 'T': T, 'U': U, 'V': V,
        'transition_matrix': np.eye(V),
        'transition_covariances': np.array(
            [make_spd_matrix(V) * 100 for k in range(K)]),
        'observation_matrix': np.eye(U),
        'observation_covariance': make_spd_matrix(U),
        'initial_state_means': np.random.random((K, V)),
        'initial_state_covariances': np.array(
            [make_spd_matrix(V) for k in range(K)]),
        'component_weights': component_weights,
        'responsibilities': responsibilities
    }

    model = LDSMixture.LDSMixture()
    model.set_initialization(initialization)
    return model


if __name__ == "__main__":

    output_directory = sys.argv[1]
    data_path = sys.argv[2]
    K = int(sys.argv[3])

    print('using:', data_path, 'to source data')

    output_directory = '/'.join(output_directory.split('/'))
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    data = pickle.load(open(data_path, 'rb'))

    N, T = data.shape[:2]
    U, V = 1, 1

    print('generating randomly initialized model')
    model = random_model(N, K, T, U, V)

    model_save_path = '/'.join(output_directory.split('/')) \
        + '/models'

    models = [model]
    pickle.dump(models, open(model_save_path, 'wb'))

    i = 0
    while(model.elbo_delta() > 1-10):
        model.estimate_states(data)
        model.estimate_responsibilities(data)
        print(model.elbo_delta())

        model.mstep(data)
        model.elbo(data)
        print(model.elbo_delta())

        models.append(copy.deepcopy(model))
        pickle.dump(models, open(model_save_path, 'wb'))
        i += 1

        if(model.elbo_history[-1] - model.elbo_history[-3] < 0):
            break

    # save final model
    model_save_path = '/'.join(output_directory.split('/')) \
        + '/model_final'
    pickle.dump(model, open(model_save_path, 'wb'))
