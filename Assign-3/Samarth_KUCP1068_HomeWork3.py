## CSP 304: Machine Learning Lab (Spring 2023)
# Homework 3

# Name: Samarth Bhatt
# College ID: 2020KUCP1068 
# Lab Batch: A3

import skimage.io as io
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import time

# function to perform EM algorithm
def EMG(image_path, K):

    # reading and reshaping the image data
    image = io.imread(image_path)
    data = image.reshape(-1,3)

    # loop over K = [4,8,12]
    for k in K:

        # initializing K-means model
        model = KMeans(n_clusters=k, max_iter=3, random_state=10)
        model.fit(data)
        cluster_assignments = model.predict(data)

        # estimating initial parameters from the model
        means = []
        covariances = []
        for i in range(k):
            cluster_pixels = data[cluster_assignments == i]
            mean = np.mean(cluster_pixels, axis=0)
            covariance = np.cov(cluster_pixels, rowvar=False)
            means.append(mean)
            covariances.append(covariance)
        priors = np.array([np.mean(cluster_assignments == i) for i in range(k)])

        # print(means)
        # print(covariances)
        # print(priors)

        # declaring and initializing posterior probability of each pixel
        h = np.zeros((data.shape[0], k))

        # declaring an array to store the log-likelihood after each iteration in EM steps
        q = []

        # setting maximum iteration to 200 and loop until it becomes 0
        max_iter = 200
        while max_iter != 0:

            # loop over k clusters
            for i in range(k):
                
                '''_E-step_'''

                # compute the posterior probablity (h) for cluster (i)
                pdf = multivariate_normal.pdf(data, means[i], covariances[i])
                h[:,i] = priors[i] * pdf

            # normalize the posterior probablity
            h /= np.sum(h, axis=1, keepdims=True)

            '''_M-step_'''

            # summation of posterior probablity of each pixel in K clusters
            h_sum = np.sum(h, axis=0)

            # estimate the new parameters
            priors = h_sum / data.shape[0]
            means = np.dot(h.T, data) / h_sum.reshape(-1, 1)
            for j in range(k):
                covariances[j] = np.dot(h[:, j] * (data - means[j]).T, data - means[j]) / h_sum[j]

            # Compute log-likelihood
            log_likelihoods = np.log(np.dot(h, priors))
            q.append(np.sum(log_likelihoods))

            max_iter -= 1

        # Display complete log-likelihood vs iteration number
        plt.plot(q)

        # Save compressed image
        compressed_image = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
        cluster_memberships = np.argmax(h, axis=1)
        for i in range(k):
            mask = cluster_memberships == i
            mask = mask.reshape(image.shape[0], image.shape[1])
            compressed_image[mask] = means[i]
        compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
        io.imsave(f'Images/compressed_{k}.png', compressed_image)
        
    plt.xlabel('Number of iteration')
    plt.ylabel('Complete Log-Likelihood')
    plt.title('Complete Log-Likelihood v/s Iterations')
    plt.show()

    return h, means, q


# main code 

K = [4,8,12]
h, m, q = EMG('Images\stadium.bmp', K)
print("h\n", h)                     # dimension = (n,k)
print("m\n", m)                     # dimension = (k,d)
print("q\n", q)                     # dimension = (1,n_iteration)
