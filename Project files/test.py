import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

data, labels, class_names, vocabulary = np.load("ReutersNews_4Classes_sparse.npy", allow_pickle=True)

def sample_indices(labels, *num_per_class):
    """
    Returns randomly selected indices. It will return the specified number of indices for each class.
    """
    indices = []
    for cls, num in enumerate(num_per_class):
        cls_indices = np.where(labels == cls)[0]
        indices.extend(np.random.choice(cls_indices, size=num, replace=False))
    return np.array(indices)


import sys
np.set_printoptions(threshold=sys.maxsize)

training_indices = sample_indices(labels, 80, 80, 80, 80)
sarray = []
for i in range(700):
    sarray.append(i)
sarray = np.array(sarray)
test_samples = sarray[~np.in1d(sarray,training_indices).reshape(sarray.shape)]
training_labels = labels[training_indices]

training_data = data[training_indices].toarray()

test_data = data[test_samples].toarray()
#pairwise_distance = np.sqrt(np.sum(np.square(training_data)[:,np.newaxis,:],axis=2)-2 * training_data @ test_samples.T + np.sum(np.square(test_samples), axis=1))


td = np.sqrt(np.sum(training_data**2,axis=1))[:,np.newaxis]
ts = np.sqrt(np.sum(test_data**2,axis=1))[np.newaxis,:]
similarity = np.dot(training_data, test_data.T)/(td * ts)
pairwise_distance = 1. - similarity

sorted_indices = np.argsort(pairwise_distance, axis=0)

nearest_indices = sorted_indices[0:5, : ]
#nearest_neighbours = data[nearest_indices]




rows, columns = nearest_indices.shape
predictions = list()
for j in range(columns):
    temp = list()
    for i in range(rows):
        cell = sorted_indices[i][j]
        temp.append(labels[cell])
    predictions.append(max(temp,key=temp.count)) #this is the key function, brings the mode value
predictions=np.array(predictions)
print(predictions)
print(labels[test_samples])