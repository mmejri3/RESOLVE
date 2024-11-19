import numpy as np
def create_sorting_dataset(objects, seqs_length, n_seqs):

    n_objects = len(objects)

    # generate random permutations of length `seqs_length` out of `vocab_size`
    seqs = np.array([np.random.choice(range(n_objects), size=seqs_length, replace=False) for _ in range(n_seqs)])
    
    # remove duplicate seqs (although very unlikely)
    _, unique_seq_idxs = np.unique(seqs, axis=0, return_inverse=True)
    seqs = seqs[unique_seq_idxs]

    # create object sequences
    object_seqs = objects[seqs]
    
    sorted_seqs = np.sort(seqs, axis=1)

    arg_sort = np.argsort(seqs, axis=1)

    
    # add `START_TOKEN` to beginning of sorting 
    start_token = seqs_length
    start_tokens = np.array([start_token] * len(arg_sort))[np.newaxis].T
    arg_sort = np.hstack([start_tokens, arg_sort])

    return seqs, sorted_seqs, arg_sort, object_seqs, start_token
    
# dataset 1
vocab_size = 64
dim = 8
seqs_length = 5
n_seqs = 10000

objects = np.random.normal(size=(vocab_size, dim))

seqs, sorted_seqs, arg_sort, object_seqs, start_token = create_sorting_dataset(objects, seqs_length, n_seqs)

target = arg_sort[:, :-1]
labels = arg_sort[:, 1:]
data = {
    'objects': objects, 'seqs': seqs, 'sorted_seqs': sorted_seqs, 'arg_sort': arg_sort,
    'object_seqs': object_seqs, 'target': target, 'labels': labels, 'start_token': start_token
    }

np.save('object_sorting_datasets/task1_object_sort_dataset.npy', data)


# dataset 2 (same paramters, just re-generate objects randomly)
vocab_size = 64
dim = 8
seqs_length = 5
n_seqs = 10000

objects = np.random.normal(size=(vocab_size, dim))

seqs, sorted_seqs, arg_sort, object_seqs, start_token = create_sorting_dataset(objects, seqs_length, n_seqs)

target = arg_sort[:, :-1]
labels = arg_sort[:, 1:]

data = {
    'objects': objects, 'seqs': seqs, 'sorted_seqs': sorted_seqs, 'arg_sort': arg_sort,
    'object_seqs': object_seqs, 'target': target, 'labels': labels, 'start_token': start_token
    }

np.save('object_sorting_datasets/task2_object_sort_dataset.npy', data)
data = np.load('object_sorting_datasets/task2_object_sort_dataset.npy', allow_pickle=True).item()
objects = data['objects']
seqs = data['seqs']

reshuffle = np.random.choice(64, size=64, replace=False)
objects_ = objects[reshuffle]
object_seqs_ = objects_[seqs]

data['reshuffle'] = reshuffle
data['objects'] = objects_
data['object_seqs'] = object_seqs_

np.save('object_sorting_datasets/task2_reshuffled_object_sort_dataset.npy', data)

# generate objects with attribute-product structure

attr1_n_objects = 4 # number of possible values for attribute 1
attr1_embedding_dim = 4 # dimension of vector representation of attribute 1 
attr2_n_objects = 12 # number of possible values for attribute 2
attr2_embedding_dim = 8 # dimension of vector representation of attribute 2

# generate vector representations of the two attributes
attr1_objects = np.random.randint(-100,100,size=(attr1_n_objects,  attr1_embedding_dim))
attr2_objects = np.random.randint(-100,100,size=(attr2_n_objects,  attr2_embedding_dim))

# generate attribute-product objects and ordering 
object_products = [(attr1, attr2) for attr1 in range(attr1_n_objects) for attr2 in range(attr2_n_objects)]

objects = []
for attr1, attr2 in object_products:
    attr1_object = attr1_objects[attr1] # get vector representation of attribute 1
    attr2_object = attr2_objects[attr2] # get vector representation of attribute 2
    object_ = np.concatenate([attr1_object, attr2_object]) # stack attributes to create object
    objects.append(object_)

objects = np.array(objects)

# generate object sorting dataset
seqs_length = 5
n_seqs = 10000

seqs, sorted_seqs, arg_sort, object_seqs, start_token = create_sorting_dataset(objects, seqs_length, n_seqs)

target = arg_sort[:, :-1]
labels = arg_sort[:, 1:]
print(target[0])
print(labels[0])
data = {
    'objects': objects, 'attr1_objects': attr1_objects, 'attr2_objects': attr2_objects, 
    'seqs': seqs, 'sorted_seqs': sorted_seqs, 'arg_sort': arg_sort,
    'object_seqs': object_seqs, 'target': target, 'labels': labels, 'start_token': start_token
    }

np.save('object_sorting_datasets/product_structure_object_sort_dataset.npy', data)

attr1_reshuffle = np.random.choice(attr1_n_objects, size=attr1_n_objects, replace=False)
attr2_reshuffle = np.arange(attr2_n_objects) # identity permutation

attr1_objects_reshuffled = attr1_objects[attr1_reshuffle]
attr2_objects_reshuffled = attr2_objects[attr2_reshuffle]

# generate attribute-product objects and ordering 
object_products = [(attr1, attr2) for attr1 in range(attr1_n_objects) for attr2 in range(attr2_n_objects)]

objects_reshuffled = []
for attr1, attr2 in object_products:
    attr1_object = attr1_objects[attr1] # get vector representation of attribute 1
    attr2_object = attr2_objects[attr2] # get vector representation of attribute 2
    object_ = np.concatenate([attr1_object, attr2_object]) # stack attributes to create object
    objects_reshuffled.append(object_)

objects_reshuffled = np.array(objects)
data['attr1_reshuffle'] = attr1_reshuffle
data['attr2_reshuffle'] = attr2_reshuffle
data['objects'] = objects_reshuffled

object_seqs_reshuffled = objects_reshuffled[seqs]
data['object_seqs'] = object_seqs_reshuffled

data['attr1_objects'] = attr1_objects_reshuffled
data['attr2_objects'] = attr2_objects_reshuffled

np.save('object_sorting_datasets/product_structure_reshuffled_object_sort_dataset.npy', data)

import numpy as np
def create_sorting_dataset(objects, seqs_length, n_seqs):

    n_objects = len(objects)

    # generate random permutations of length `seqs_length` out of `vocab_size`
    seqs = np.array([np.random.choice(range(n_objects), size=seqs_length, replace=False) for _ in range(n_seqs)])
    
    # remove duplicate seqs (although very unlikely)
    _, unique_seq_idxs = np.unique(seqs, axis=0, return_inverse=True)
    seqs = seqs[unique_seq_idxs]

    # create object sequences
    object_seqs = objects[seqs]
    
    sorted_seqs = np.sort(seqs, axis=1)

    arg_sort = np.argsort(seqs, axis=1)

    
    # add `START_TOKEN` to beginning of sorting 
    start_token = seqs_length
    start_tokens = np.array([start_token] * len(arg_sort))[np.newaxis].T
    arg_sort = np.hstack([start_tokens, arg_sort])

    return seqs, sorted_seqs, arg_sort, object_seqs, start_token

# dataset 1
vocab_size = 64
dim = 8
seqs_length = 5
n_seqs = 10000

objects = np.random.normal(size=(vocab_size, dim))

seqs, sorted_seqs, arg_sort, object_seqs, start_token = create_sorting_dataset(objects, seqs_length, n_seqs)

target = arg_sort[:, :-1]
labels = arg_sort[:, 1:]
data = {
    'objects': objects, 'seqs': seqs, 'sorted_seqs': sorted_seqs, 'arg_sort': arg_sort,
    'object_seqs': object_seqs, 'target': target, 'labels': labels, 'start_token': start_token
    }

np.save('object_sorting_datasets/task1_object_sort_dataset.npy', data)
# dataset 2 (same paramters, just re-generate objects randomly)
vocab_size = 64
dim = 8
seqs_length = 6
n_seqs = 10000

objects = np.random.normal(size=(vocab_size, dim))

seqs, sorted_seqs, arg_sort, object_seqs, start_token = create_sorting_dataset(objects, seqs_length, n_seqs)

target = arg_sort[:, :-1]
labels = arg_sort[:, 1:]

data = {
    'objects': objects, 'seqs': seqs, 'sorted_seqs': sorted_seqs, 'arg_sort': arg_sort,
    'object_seqs': object_seqs, 'target': target, 'labels': labels, 'start_token': start_token
    }

np.save('object_sorting_datasets/task2_object_sort_dataset.npy', data)
data = np.load('object_sorting_datasets/task2_object_sort_dataset.npy', allow_pickle=True).item()
objects = data['objects']
seqs = data['seqs']

reshuffle = np.random.choice(64, size=64, replace=False)
objects_ = objects[reshuffle]
object_seqs_ = objects_[seqs]

data['reshuffle'] = reshuffle
data['objects'] = objects_
data['object_seqs'] = object_seqs_

np.save('object_sorting_datasets/task2_reshuffled_object_sort_dataset.npy', data)
# generate objects with attribute-product structure

attr1_n_objects = 4 # number of possible values for attribute 1
attr1_embedding_dim = 4 # dimension of vector representation of attribute 1 
attr2_n_objects = 12 # number of possible values for attribute 2
attr2_embedding_dim = 8 # dimension of vector representation of attribute 2

# generate vector representations of the two attributes
attr1_objects = np.random.randint(-100,100,size=(attr1_n_objects,  attr1_embedding_dim))
attr2_objects = np.random.randint(-100,100,size=(attr2_n_objects,  attr2_embedding_dim))

# generate attribute-product objects and ordering 
object_products = [(attr1, attr2) for attr1 in range(attr1_n_objects) for attr2 in range(attr2_n_objects)]

objects = []
for attr1, attr2 in object_products:
    attr1_object = attr1_objects[attr1] # get vector representation of attribute 1
    attr2_object = attr2_objects[attr2] # get vector representation of attribute 2
    object_ = np.concatenate([attr1_object, attr2_object]) # stack attributes to create object
    objects.append(object_)

objects = np.array(objects)
# generate object sorting dataset
seqs_length = 5
n_seqs = 10000

seqs, sorted_seqs, arg_sort, object_seqs, start_token = create_sorting_dataset(objects, seqs_length, n_seqs)

target = arg_sort[:, :-1]
labels = arg_sort[:, 1:]
print(target[0])
print(labels[0])
data = {
    'objects': objects, 'attr1_objects': attr1_objects, 'attr2_objects': attr2_objects, 
    'seqs': seqs, 'sorted_seqs': sorted_seqs, 'arg_sort': arg_sort,
    'object_seqs': object_seqs, 'target': target, 'labels': labels, 'start_token': start_token
    }

np.save('object_sorting_datasets/product_structure_object_sort_dataset.npy', data)
attr1_reshuffle = np.random.choice(attr1_n_objects, size=attr1_n_objects, replace=False)
attr2_reshuffle = np.arange(attr2_n_objects) # identity permutation

attr1_objects_reshuffled = attr1_objects[attr1_reshuffle]
attr2_objects_reshuffled = attr2_objects[attr2_reshuffle]

# generate attribute-product objects and ordering 
object_products = [(attr1, attr2) for attr1 in range(attr1_n_objects) for attr2 in range(attr2_n_objects)]

objects_reshuffled = []
for attr1, attr2 in object_products:
    attr1_object = attr1_objects[attr1] # get vector representation of attribute 1
    attr2_object = attr2_objects[attr2] # get vector representation of attribute 2
    object_ = np.concatenate([attr1_object, attr2_object]) # stack attributes to create object
    objects_reshuffled.append(object_)

objects_reshuffled = np.array(objects)
data['attr1_reshuffle'] = attr1_reshuffle
data['attr2_reshuffle'] = attr2_reshuffle
data['objects'] = objects_reshuffled

object_seqs_reshuffled = objects_reshuffled[seqs]
data['object_seqs'] = object_seqs_reshuffled

data['attr1_objects'] = attr1_objects_reshuffled
data['attr2_objects'] = attr2_objects_reshuffled

np.save('object_sorting_datasets/product_structure_reshuffled_object_sort_dataset.npy', data)


