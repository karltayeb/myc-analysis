import sys
import os
import pickle
import numpy as np
import copy
from LDSMixture import LDSMixture


if __name__ == "__main__":

    output_directory = sys.argv[1]
    data_path = sys.argv[2]
    initialization_path = sys.argv[3]
    threshold = float(sys.argv[4])

    print('using:', data_path, 'to source data')
    print('using:', initialization_path, 'for model parameters')
    print('declaring convergence at elbo change < ', int(threshold))

    output_directory = '/'.join(output_directory.split('/'))
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    data = pickle.load(open(data_path, 'rb'))
    initialization = pickle.load(open(initialization_path, 'rb'))

    model = LDSMixture()
    model.set_initialization(initialization)

    N = data.shape[0]
    K = model.K

    print('generating random initial responsibilities')
    responsibilities = np.random.uniform(5, 10, (N, K))
    responsibilities = \
        responsibilities / responsibilities.sum(axis=1)[:, np.newaxis]

    model.responsibilities = responsibilities

    model_save_path = '/'.join(output_directory.split('/')) \
        + '/models'

    models = [model]
    pickle.dump(models, open(model_save_path, 'wb'))
    i = 0
    while(model.elbo_delta() > threshold):
        model.estimate_states(data)
        model.elbo(data)
        print(model.elbo_delta())
        if (model.elbo_delta() < 0):
            break

        model.estimate_responsibilities(data)
        model.elbo(data)
        print(model.elbo_delta())
        if(model.elbo_delta() < 0):
            break

        models.append(copy.deepcopy(model))
        pickle.dump(models, open(model_save_path, 'wb'))
        i += 1

    # save final model
    model_save_path = '/'.join(output_directory.split('/')) \
        + '/model_final'
    pickle.dump(model, open(model_save_path, 'wb'))
