import sys

PATH_WORKDIR = "/home/khaldi/Documents/text-clustering/"
sys.path.append(PATH_WORKDIR)
from globals import PARAMS
import os
import math
import numpy as np
import tensorflow as tf
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from dkm_work.utils import cluster_acc
from dkm_work.utils import next_batch
from dkm_work.compgraph import DkmCompGraph

tf.compat.v1.disable_eager_execution()


def deep_kmeans(dataset="CLASSIC3", params: dict = PARAMS, n_runs=1) -> dict:
    """
        input:
            -dataset belong to {"CLASSIC3","CLASSIC4","BBC"}
            -params: parameter of DeepKMeans (see globals file and repo:
            "https://github.com/MaziarMF/deep-k-means")
        return:
            dictionnary of results (depends on the params input)
    """

    # Dataset setting from arguments
    if dataset == "CLASSIC3":
        from dkm_work.split.classic3 import classic3_specs as specs
    elif dataset == "CLASSIC4":
        from dkm_work.split.classic4 import classic4_specs as specs
    elif dataset == "BBC":
        from dkm_work.split.bbc import bbc_specs as specs

    else:
        raise ValueError(f"Unknown dataset: {dataset}!")

    # Parameter setting from arguments
    n_pretrain_epochs = params["p_epochs"]
    n_finetuning_epochs = params["f_epochs"]
    lambda_ = params["lambda"]
    batch_size = params["batch_size"]  # Size of the mini-batches used in the stochastic optimizer
    n_batches = int(math.ceil(specs.n_samples / batch_size))  # Number of mini-batches
    validation = params["validation"]  # Specify if data should be split into validation and test
    # sets
    pretrain = params["pretrain"]  # Specify if DKM's autoencoder should be pretrained
    annealing = params["annealing"]  # Specify if annealing should be used
    seeded = params["seeded"]  # Specify if runs are seeded

    cpu = params["cpu"]

    print("Hyperparameters...")
    print("lambda =", lambda_)

    # Define the alpha scheme depending on if the approach includes annealing/pretraining
    if annealing and not pretrain:
        constant_value = 1  # specs.embedding_size # Used to modify the range of the alpha scheme
        max_n = 40  # Number of alpha values to consider
        alphas = np.zeros(max_n, dtype=float)
        alphas[0] = 0.1
        for i in range(1, max_n):
            alphas[i] = (2 ** (1 / (np.log(i + 1)) ** 2)) * alphas[i - 1]
        alphas = alphas / constant_value
    elif not annealing and pretrain:
        constant_value = 1  # specs.embedding_size # Used to modify the range of the alpha scheme
        max_n = 20  # Number of alpha values to consider (constant values are used here)
        alphas = 1000 * np.ones(max_n, dtype=float)  # alpha is constant
        alphas = alphas / constant_value
    else:
        print("Run with either annealing  or pretraining , but not both.")
        exit()

    # Select only the labels which are to be used in the evaluation (disjoint for validation and
    # test)
    if validation:
        validation_target = np.asarray([specs.target[i] for i in specs.validation_indices])
        test_target = np.asarray([specs.target[i] for i in specs.test_indices])
    else:
        target = specs.target

    # Dataset on which the computation graph will be run
    data = specs.data

    # Hardware specifications
    if cpu:
        os.environ[
            'CUDA_VISIBLE_DEVICES'] = '-1'  # Run on CPU instead of GPU if batch_size is small
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.05)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

    # Definition of the randomly-drawn (0-10000) seeds to be used for each run
    seeds = list(np.random.randint(low=0, high=10001, size=n_runs))
    # seeds = [8905, 9129, 291, 4012, 1256, 6819, 4678, 6971, 1362, 575]

    if validation:
        list_validation_acc = []
        list_validation_ari = []
        list_validation_nmi = []
        list_validation_embeddings = []
        list_test_acc = []
        list_test_ari = []
        list_test_nmi = []
        list_test_embeddings = []
    else:
        list_acc = []
        list_ari = []
        list_nmi = []
        list_embeddings = []

    for run in range(n_runs):
        # Use a fixed seed for this run, as defined in the seed list
        if seeded:
            tf.compat.v1.reset_default_graph()
            tf.compat.v1.set_random_seed(seeds[run])
            np.random.seed(seeds[run])

        print("Run", run)

        # Define the computation graph for DKM
        cg = DkmCompGraph([specs.dimensions, specs.activations, specs.names], specs.n_clusters,
                          lambda_)

        # Run the computation graph
        with tf.compat.v1.Session(config=config) as sess:
            # Initialization
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

            # Variables to save tensor content
            distances = np.zeros((specs.n_clusters, specs.n_samples))
            embeddings = np.zeros((specs.n_samples, specs.embedding_size), dtype=float)

            # Pretrain if specified
            if pretrain:
                print("Starting autoencoder pretraining...")

                # Variables to save pretraining tensor content
                embeddings = np.zeros((specs.n_samples, specs.embedding_size), dtype=float)

                # First, pretrain the autoencoder
                # Loop over epochs
                for epoch in range(n_pretrain_epochs):
                    print("Pretraining step: epoch {}".format(epoch))

                    # Loop over the samples
                    for _ in range(n_batches):
                        # Fetch a random data batch of the specified size
                        indices, data_batch = next_batch(batch_size, data)

                        # Run the computation graph until pretrain_op (only on autoencoder) on
                        # the data batch
                        _, embedding_, ae_loss_ = sess.run(
                            (cg.pretrain_op, cg.embedding, cg.ae_loss),
                            feed_dict={cg.input: data_batch})

                        # Save the embeddings for batch samples
                        for j in range(len(indices)):
                            embeddings[indices[j], :] = embedding_[j, :]

                        # print("ae_loss_:", float(ae_loss_))

                # Second, run k-means++ on the pretrained embeddings
                print("Running k-means on the learned embeddings...")
                kmeans_model = KMeans(n_clusters=specs.n_clusters, init="k-means++").fit(embeddings)

                if validation:
                    # Split the cluster assignments into validation and test ones
                    validation_cluster_assign = np.asarray(
                        [kmeans_model.labels_[i] for i in specs.validation_indices])
                    test_cluster_assign = np.asarray(
                        [kmeans_model.labels_[i] for i in specs.test_indices])
                    # Evaluate the clustering validation performance using the ground-truth labels
                    validation_acc = cluster_acc(validation_target, validation_cluster_assign)
                    print("Validation ACC", validation_acc)
                    validation_ari = adjusted_rand_score(validation_target,
                                                         validation_cluster_assign)
                    print("Validation ARI", validation_ari)
                    validation_nmi = normalized_mutual_info_score(validation_target,
                                                                  validation_cluster_assign)
                    print("Validation NMI", validation_nmi)
                    # Evaluate the clustering test performance using the ground-truth labels
                    test_acc = cluster_acc(test_target, test_cluster_assign)
                    print("Test ACC", test_acc)
                    test_ari = adjusted_rand_score(test_target, test_cluster_assign)
                    print("Test ARI", test_ari)
                    test_nmi = normalized_mutual_info_score(test_target, test_cluster_assign)
                    print("Test NMI", test_nmi)

                else:
                    # Evaluate the clustering performance using the ground-truth labels
                    acc = cluster_acc(target, kmeans_model.labels_)
                    print("ACC", acc)
                    ari = adjusted_rand_score(target, kmeans_model.labels_)
                    print("ARI", ari)
                    nmi = normalized_mutual_info_score(target, kmeans_model.labels_)
                    print("NMI", nmi)

                # The cluster centers are used to initialize the cluster representatives in DKM
                sess.run(tf.compat.v1.assign(cg.cluster_rep, kmeans_model.cluster_centers_))

            # Train the full DKM model
            if len(alphas) > 0:
                print("Starting DKM training...")
            # Loop over alpha (inverse temperature), from small to large values
            for k in range(len(alphas)):
                print("Training step: alpha[{}]: {}".format(k, alphas[k]))

                # Loop over epochs per alpha
                for _ in range(n_finetuning_epochs):
                    # Loop over the samples
                    for _ in range(n_batches):
                        # print("Training step: alpha[{}], epoch {}".format(k, i))

                        # Fetch a random data batch of the specified size
                        indices, data_batch = next_batch(batch_size, data)

                        # print(tf.trainable_variables()) current_batch_size = np.shape(
                        # data_batch)[0] # Can be different from batch_size for unequal splits

                        # Run the computation graph on the data batch
                        _, loss_, stack_dist_, cluster_rep_, ae_loss_, kmeans_loss_, embeddings_ = \
                            sess.run((cg.train_op, cg.loss, cg.stack_dist, cg.cluster_rep,
                                      cg.ae_loss, cg.kmeans_loss, cg.embedding),
                                     feed_dict={cg.input: data_batch, cg.alpha: alphas[k]})

                        # Save the embeddings for batch samples
                        for j in range(len(indices)):
                            embeddings[indices[j], :] = embeddings_[j, :]

                        # Save the distances for batch samples
                        for j in range(len(indices)):
                            distances[:, indices[j]] = stack_dist_[:, j]

                # Evaluate the clustering performance every print_val alpha and for last alpha
                print_val = 1
                if k % print_val == 0 or k == max_n - 1:
                    print("loss:", loss_)
                    print("ae loss:", ae_loss_)
                    print("kmeans loss:", kmeans_loss_)

                    # Infer cluster assignments for all samples
                    cluster_assign = np.zeros(specs.n_samples, dtype=float)
                    for i in range(specs.n_samples):
                        index_closest_cluster = np.argmin(distances[:, i])
                        cluster_assign[i] = index_closest_cluster
                    cluster_assign = cluster_assign.astype(np.int64)

                    if validation:
                        validation_cluster_assign = np.asarray(
                            [cluster_assign[i] for i in specs.validation_indices])
                        test_cluster_assign = np.asarray(
                            [cluster_assign[i] for i in specs.test_indices])

                        # Evaluate the clustering validation performance using the ground-truth
                        # labels
                        validation_acc = cluster_acc(validation_target, validation_cluster_assign)
                        print("Validation ACC", validation_acc)
                        validation_ari = adjusted_rand_score(validation_target,
                                                             validation_cluster_assign)
                        print("Validation ARI", validation_ari)
                        validation_nmi = normalized_mutual_info_score(validation_target,
                                                                      validation_cluster_assign)
                        print("Validation NMI", validation_nmi)

                        # Evaluate the clustering test performance using the ground-truth labels
                        test_acc = cluster_acc(test_target, test_cluster_assign)
                        print("Test ACC", test_acc)
                        test_ari = adjusted_rand_score(test_target, test_cluster_assign)
                        print("Test ARI", test_ari)
                        test_nmi = normalized_mutual_info_score(test_target, test_cluster_assign)
                        print("Test NMI", test_nmi)
                        validation_embeddings = np.array(
                            [embeddings[i, :] for i in specs.validation_indices])
                        test_embeddings = np.array([embeddings[i, :] for i in specs.test_indices])
                    else:
                        # Evaluate the clustering performance using the ground-truth labels
                        acc = cluster_acc(target, cluster_assign)
                        print("ACC", acc)
                        ari = adjusted_rand_score(target, cluster_assign)
                        print("ARI", ari)
                        nmi = normalized_mutual_info_score(target, cluster_assign)
                        print("NMI", nmi)

            # Record the clustering performance for the run
            if validation:
                list_validation_acc.append(validation_acc)
                list_validation_ari.append(validation_ari)
                list_validation_nmi.append(validation_nmi)
                list_validation_embeddings.append(validation_embeddings)
                list_test_acc.append(test_acc)
                list_test_ari.append(test_ari)
                list_test_nmi.append(test_nmi)
                list_test_embeddings.append(test_embeddings)
            else:
                list_acc.append(acc)
                list_ari.append(ari)
                list_nmi.append(nmi)
                list_embeddings.append(embeddings)

    if validation:
        list_validation_acc = np.array(list_validation_acc)
        print("Average validation ACC: {:.3f} +/- {:.3f}".format(np.mean(list_validation_acc),
                                                                 np.std(list_validation_acc)))
        list_validation_ari = np.array(list_validation_ari)
        print("Average validation ARI: {:.3f} +/- {:.3f}".format(np.mean(list_validation_ari),
                                                                 np.std(list_validation_ari)))
        list_validation_nmi = np.array(list_validation_nmi)
        print("Average validation NMI: {:.3f} +/- {:.3f}".format(np.mean(list_validation_nmi),
                                                                 np.std(list_validation_nmi)))

        list_test_acc = np.array(list_test_acc)
        print("Average test ACC: {:.3f} +/- {:.3f}".format(np.mean(list_test_acc),
                                                           np.std(list_test_acc)))
        list_test_ari = np.array(list_test_ari)
        print("Average test ARI: {:.3f} +/- {:.3f}".format(np.mean(list_test_ari),
                                                           np.std(list_test_ari)))
        list_test_nmi = np.array(list_test_nmi)
        print("Average test NMI: {:.3f} +/- {:.3f}".format(np.mean(list_test_nmi),
                                                           np.std(list_test_nmi)))
        return {
            "validation(train)": {
                "labels": validation_cluster_assign,
                "acc": list_validation_acc,
                "ari": list_validation_ari,
                "nmi": list_validation_nmi,
                "embeddings": list_validation_embeddings
            },
            "test": {
                "labels": test_cluster_assign,
                "acc": list_test_acc,
                "ari": list_test_ari,
                "nmi": list_test_nmi,
                "embeddings": list_test_embeddings

            }
        }
    else:
        list_acc = np.array(list_acc)
        print("Average ACC: {:.3f} +/- {:.3f}".format(np.mean(list_acc), np.std(list_acc)))
        list_ari = np.array(list_ari)
        print("Average ARI: {:.3f} +/- {:.3f}".format(np.mean(list_ari), np.std(list_ari)))
        list_nmi = np.array(list_nmi)
        print("Average NMI: {:.3f} +/- {:.3f}".format(np.mean(list_nmi), np.std(list_nmi)))
        return {
            "labels": cluster_assign,
            "acc": list_acc,
            "ari": list_ari,
            "nmi": list_nmi,
            "embeddings": embeddings
        }


if __name__ == "__main__":
    print(deep_kmeans(dataset="CLASSIC4", params=PARAMS, n_runs=1))
