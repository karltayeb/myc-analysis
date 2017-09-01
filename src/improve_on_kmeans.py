import numpy as np
import copy
import pickle
import sys
import os

from sklearn.cluster import KMeans

from LDSMixture import LDSMixture


def learn_states(model, data, output_directory, threshold):
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    model_save_path = '/'.join(output_directory.split('/')) \
        + '/models'

    models = [model]
    pickle.dump(models, open(model_save_path, 'wb'))

    i = 0
    while(model.elbo_delta() > threshold):
        model.estimate_states(data)
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

if __name__ == "__main__":

    basedir = sys.argv[1]
    output_id = sys.argv[2]
    init_type = sys.argv[3]

    data_path = basedir + '/data'
    truth_path = basedir + '/truth'

    data = pickle.load(open(data_path, 'rb'))
    truth = pickle.load(open(truth_path, 'rb'))

    K = truth['K']
    N = data.shape[0]

    basedir = '/'.join(basedir.split('/'))

    if init_type == 'kmeans':
        out_dir = basedir + 'k-' + output_id

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        kmeans = KMeans(n_clusters=K).fit(data)
        kmeans_path = out_dir + '/kmeans'

        pickle.dump(kmeans, open(kmeans_path, 'wb'))

        model = LDSMixture()
        model.set_initialization(truth)

        responsibilities = np.zeros((N, K))
        for n in range(N):
            k = kmeans.labels_[n]
            responsibilities[n, k] = 1

        model.responsibilities = responsibilities
        learn_states(model, data, out_dir, 1e-5)

    if init_type == 'random':
        model = LDSMixture()
        model.set_initialization(truth)

        out_dir = basedir + 'r-' + output_id

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        responsibilities = np.zeros((N, K))
        for n in range(N):
            k = np.random.choice(a=K, p=model.component_weights)
            responsibilities[n, k] = 1

        model.responsibilities = responsibilities
        learn_states(model, data, out_dir, 1e-5)
