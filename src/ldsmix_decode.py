import sys
import os
import pickle
import numpy as np
from LDSMixture import LDSMixture


if __name__ == "__main__":

    output_directory = sys.argv[1]
    data_path = sys.argv[2]
    initialization_path = sys.argv[3]
    threshold = float(sys.argv[4])
    processes = int(sys.argv[5])

    output_directory = '/'.join(output_directory.split('/'))
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    data = pickle.load(open(data_path, 'rb'))
    initialization = pickle.load(open(initialization_path, 'rb'))

    model = LDSMixture()
    model.set_initialization(initialization)

    N = data.shape[0]
    K = model.K

    responsibilities = np.random.uniform(0, 1, (N, K))
    responsibilities = \
        responsibilities / responsibilities.sum(axis=1)[:, np.newaxis]

    model.responsibilities = responsibilities
    models = [model]
    elbos = [-1e100]
    i = 0
    while(True):
        model_save_path = '/'.join(output_directory.split('/')) \
            + '/model' + str(i)
        pickle.dump(model, open(model_save_path, 'wb'))

        i += 1
        print(i)
        print('Estimating states')
        model.estimate_states(data, processes=processes)
        print('Estimating responsibilities')
        model.estimate_responsibilities(data, processes=processes)
        print('Computing elbo')
        model.elbo(data)

        elbos.append(model.elbo_history[-1])
        diff = elbos[-1] - elbos[-2]
        print(i, diff)
        if (diff) < threshold:
            break

    model_save_path = '/'.join(output_directory.split('/')) \
        + '/model_final'

    pickle.dump(model, open(model_save_path, 'wb'))
