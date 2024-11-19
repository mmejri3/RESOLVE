from transformer_modules import Encoder, Decoder, AddPositionalEmbedding
from abstracters import SymbolicAbstracter, RelationalAbstracter
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

import itertools

import utils

from tqdm import tqdm, trange
import wandb

import plotly.express as px

import sys
sys.path.append('../..'); sys.path.append('../');

# create data set for learning order relatios
#num_objects = 32
#object_dim = 8
num_objects = 64
object_dim = 32
object_pairs = np.array(list(itertools.permutations(range(num_objects), r=2)))
object_order_relations = (object_pairs[:,0] < object_pairs[:, 1]).astype(int)
sample = np.random.choice(len(object_pairs), 10)
for object_pair, relation in zip(object_pairs[sample], object_order_relations[sample]):
    print(f'object pair: {tuple(object_pair)}; relation: {relation}')
    
objects = np.random.normal(loc=0., scale=1., size=(num_objects, object_dim))

X = objects[object_pairs]
y = object_order_relations



from sklearn.model_selection import train_test_split
test_size = 0.35
val_size = 0.15
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size))


loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='binary_crossentropy')
create_opt = lambda : tf.keras.optimizers.AdamW(1e-3)

from abstractor import Abstractor
from abstracters import RelationalAbstracter
from tensorflow.keras import layers


import tensorflow as tf
from tensorflow.keras import layers, models
# Define the MLP used to process the input feature vectors
def build_mlp(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(128, input_shape=input_shape))
    model.add(layers.Dense(64,))
    model.add(layers.Dense(32))
    return model
    
def build_len_binary(input_shape):
    # MLP model to process each of the input feature vectors (shape: (320,))
    mlp_model = build_mlp((32,))
    
    # Input for the single feature vector (shape: (3, 320))
    input_tensor = layers.Input(shape=(2, 32))
    
    # Split the input tensor into 3 separate tensors of shape (320,)
    input1, input2 = tf.split(input_tensor, num_or_size_splits=2, axis=1)
    
    # Squeeze the tensors to remove extra dimensions (from (1, 320) to (320,))
    input1 = tf.squeeze(input1, axis=1)
    input2 = tf.squeeze(input2, axis=1)

    # Compute embeddings for each of the input vectors
    embedding1 = mlp_model(input1)
    embedding2 = mlp_model(input2)
    
    # Concatenate embeddings and pass them through another dense layer
    concatenated = layers.Concatenate()([embedding1, embedding2])
    reasoning_output = layers.Dense(16, activation='relu')(concatenated)
    #reasoning_output = layers.Dense(32, activation='relu')(reasoning_output)
    
    # Output layer with sigmoid activation for binary classification
    output = layers.Dense(1, activation='sigmoid')(reasoning_output)
    
    # Build the model
    model = models.Model(inputs=input_tensor, outputs=output)
    return model

# Example input shape and number of feature vectors
input_shape = (2, 32)  # 3 feature vectors of size 320 each
len_binary_model = build_len_binary(input_shape)

from tensorflow.keras import layers
from baseline_models.predinet import PrediNet

embedding_dim = 64
predinet_kwargs = dict(embedding_dim=embedding_dim, predinet_kwargs=dict(key_dim=4, n_heads=4, n_relations=16, add_temp_tag=False))

class PrediNetModel(tf.keras.Model):
    def __init__(self, embedding_dim, predinet_kwargs, name=None):
        super().__init__(name=name)
        self.flatten = layers.Flatten()
        self.predinet = PrediNet(**predinet_kwargs)
        self.flatten = layers.Flatten()
        self.hidden_dense = layers.Dense(32, activation='relu', name='hidden_layer')
        self.final_layer = layers.Dense(1, activation='sigmoid', name='final_layer')

    def call(self, inputs):
        x = self.predinet(inputs)
        x = self.flatten(x)
        x = self.hidden_dense(x)
        x = self.final_layer(x)
        return x

    def print_summary(self, input_shape):
        inputs = layers.Input(input_shape)
        outputs = self.call(inputs)
        print(tf.keras.Model(inputs, outputs).summary())

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='binary_crossentropy')
create_opt = lambda : tf.keras.optimizers.Adam(learning_rate=0.001)

predinet_model = PrediNetModel(**predinet_kwargs, name='predinet')
predinet_model.compile(loss='binary_crossentropy', optimizer=create_opt(), metrics=['binary_accuracy']);
predinet_model(X_train[:32]);
predinet_model.print_summary(X.shape[1:]);


from multi_head_relation import MultiHeadRelation
from tensorflow.keras import layers

embedding_dim = 64
ordercorelnet_kwargs = dict(embedding_dim=embedding_dim)

class CorelNetModel(tf.keras.Model):
    def __init__(self, embedding_dim, name=None):
        super().__init__(name=name)
        self.embedder = layers.Dense(embedding_dim)
        self.mhr = MultiHeadRelation(rel_dim=1, proj_dim=None, symmetric=True, dense_kwargs=dict(use_bias=False))
        self.flatten = layers.Flatten()
        self.hidden_dense = layers.Dense(32, activation='relu', name='hidden_layer')
        self.final_layer = layers.Dense(1, activation='sigmoid', name='final_layer')

    def call(self, inputs):
        x = self.embedder(inputs)
        x = self.mhr(x)
        x = tf.nn.softmax(x, axis=1)
        x = self.flatten(x)
        x = self.hidden_dense(x)
        x = self.final_layer(x)

        return x

    def print_summary(self, input_shape):
        inputs = layers.Input(input_shape)
        outputs = self.call(inputs)
        print(tf.keras.Model(inputs, outputs).summary())

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='binary_crossentropy')
create_opt = lambda : tf.keras.optimizers.Adam(learning_rate=0.001)

corelnet_model = CorelNetModel(**ordercorelnet_kwargs, name='order_corelnet')
corelnet_model.compile(loss='binary_crossentropy', optimizer=create_opt(), metrics=['binary_accuracy'])
corelnet_model(X_train[:32]);
corelnet_model.print_summary(X.shape[1:]);
    
class ConvLayerWithLearnableH(tf.keras.layers.Layer):
    def __init__(self, h_size):
        super(ConvLayerWithLearnableH, self).__init__()
        # Initialize h as a trainable variable
        self.h_size = h_size
    def build(self, input_shape):
        self.h = self.add_weight(shape=(2,self.h_size), initializer='random_normal', trainable=True)
    def call(self, x_batch):
        S = 2

        def convolve_step(i):
            #h_flipped = tf.reverse(self.h[i,:], axis=[0])
            h_flipped = self.h[i,:]
            padded_x_batch = tf.pad(x_batch[:, i, :], [[0, 0], [960, 960]])
            x_batch_reshaped = tf.reshape(padded_x_batch, [tf.shape(padded_x_batch)[0], -1, 1])  # Shape: (batch_size, signal_length, 1)
            h_reshaped = tf.reshape(h_flipped, [-1, 1, 1])  # Reshape filter (kernel) after flipping
            y_batch = tf.nn.conv1d(x_batch_reshaped, h_reshaped, stride=1, padding='VALID')
            return tf.squeeze(y_batch)
        output = tf.map_fn(convolve_step, tf.range(S), dtype=tf.float32)
        output = tf.transpose(output, perm=[1, 0, 2])

        return output


class SmallConvNet(tf.keras.Model):
    def __init__(self):
        super(SmallConvNet, self).__init__()
        self.encoder1 = ConvLayerWithLearnableH( h_size=1024-64+1)
        self.dropout = layers.Dropout(0.1)
        self.ln1 = layers.LayerNormalization()

    def call(self, x):
        #x = self.conv1(x)
        x = self.encoder1(x) 
        x = self.ln1(x)
        x = tf.nn.silu(x)    
        x = self.dropout(x, training=True)
        return x

class HDSymbolicAttention(layers.Layer):
    def __init__(self, d_model, embd_size, symbolic, name="hd_symbolic_attention", **kwargs):
        super(HDSymbolicAttention, self).__init__(name=name, **kwargs)
        self.d_model = d_model
        self.embd_size = embd_size
        self.symbolic = symbolic
        self.process = SmallConvNet()
        
    def cosine_similarity(self, a, b):
        dot_product = tf.reduce_mean(tf.math.sign(a) * tf.math.sign(b), axis=-1) 
        return dot_product

    def create_cosine_similarity_matrix(self, X,C):
        X_i_expanded = X[:, tf.newaxis, :, :]  
        X_j_expanded = C[:, :, tf.newaxis, :]  
        X_i_plus_X_j = X_i_expanded + X_j_expanded  

        S = self.cosine_similarity(X_i_expanded, X_i_plus_X_j)
          
        return tf.nn.softmax(S,axis=-1)

    def build(self, input_shape):
        normal_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.Symbols = tf.Variable(normal_initializer(shape=(2, self.embd_size)), name='symbols', trainable=True)
        super(HDSymbolicAttention, self).build(input_shape)

    def call(self, values):
        self.S3 = tf.zeros_like(values)
        values_projected = self.process(values)
        symbol_projected = self.process(self.S3 +self.Symbols)   
        if self.symbolic:
            scores = self.create_cosine_similarity_matrix(values_projected,symbol_projected)
        else:
            scores = self.create_cosine_similarity_matrix(values_projected,values_projected) 
        attention_output = tf.matmul(scores,values_projected)
        O = tf.nn.swish(attention_output*symbol_projected)
        return O

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.d_model)


class RESOLVEModel(tf.keras.Model):
    def __init__(self, embedding_dim, name=None):
        super().__init__(name=name)
        self.embedder = layers.Dense(embedding_dim)
        self.nvsa = HDSymbolicAttention(1024,64,False)
        self.flatten = layers.Flatten()
        self.final_layer = layers.Dense(1, activation='sigmoid', name='final_layer')

    def call(self, inputs):
        x = self.embedder(inputs)
        x = self.nvsa(x)
        x = self.flatten(x)
        x = self.final_layer(x)

        return x

    def print_summary(self, input_shape):
        inputs = layers.Input(input_shape)
        outputs = self.call(inputs)
        print(tf.keras.Model(inputs, outputs, name=self.name).summary())


class TransformerOrderModel(tf.keras.Model):
    def __init__(self, embedding_dim, encoder_kwargs, name=None):
        super().__init__(name=name)
        self.embedder = layers.Dense(embedding_dim)
        self.encoder = Encoder(**encoder_kwargs, name='encoder')
        #self.hidden_dense = layers.Dense(32, activation='relu', name='hidden_layer')
        self.flatten = layers.Flatten()
        self.final_layer = layers.Dense(1, activation='sigmoid', name='final_layer')

    def call(self, inputs):
        x = self.embedder(inputs)
        x = self.encoder(x)
        x = self.flatten(x)
        #x = self.hidden_dense(x)
        x = self.final_layer(x)

        return x

    def print_summary(self, input_shape):
        inputs = layers.Input(input_shape)
        outputs = self.call(inputs)
        print(tf.keras.Model(inputs, outputs, name=self.name).summary())
        
        
        
        
#--------------MLP------------------------------------------
class MLPModel(tf.keras.Model):
    def __init__(self, embedding_dim, h1, h2, name=None):
        super().__init__(name=name)
        self.flatten = layers.Flatten()
        self.hidden_dense1 = layers.Dense(h1, activation='relu', name='hidden_layer1')
        self.hidden_dense2 = layers.Dense(h2, activation='relu', name='hidden_layer2')
        self.final_layer = layers.Dense(1, activation='sigmoid', name='final_layer')

    def call(self, inputs):
        # x = self.embedder(inputs)
        x = inputs
        x = self.flatten(x)
        x = self.hidden_dense1(x)
        x = self.hidden_dense2(x)
        x = self.final_layer(x)

        return x

    def print_summary(self, input_shape):
        inputs = layers.Input(input_shape)
        outputs = self.call(inputs)
        print(tf.keras.Model(inputs, outputs).summary())
#-------------Abstractor---------------------------------
class AbstractorOrderModel(tf.keras.Model):
    def __init__(self, embedding_dim, abstractor_kwargs, name=None):
        super().__init__(name=name)
        self.embedder = layers.Dense(embedding_dim)
        self.abstractor = RelationalAbstracter(**abstractor_kwargs)
        self.flatten = layers.Flatten()
        self.hidden_dense = layers.Dense(32, activation='relu', name='hidden_layer')
        self.final_layer = layers.Dense(1, activation='sigmoid', name='final_layer')

    def call(self, inputs):
        x = self.embedder(inputs)
        x = self.abstractor(x)
        x = self.flatten(x)
        x = self.hidden_dense(x)
        x = self.final_layer(x)

        return x

    def print_summary(self, input_shape):
        inputs = layers.Input(input_shape)
        outputs = self.call(inputs)
        print(tf.keras.Model(inputs, outputs, name=self.name).summary())


import gc
import numpy as np
import copy
import random
from sklearn.utils import shuffle
accuracies = []
seeds = [2020, 2021, 2022]

embedding_dim = 64
mlp_kwargs = dict(embedding_dim=embedding_dim, h1=16, h2=16)
abstractor_kwargs = dict(num_layers=1, num_heads=2, dff=64,
     use_pos_embedding=False, mha_activation_type='softmax', dropout_rate=0.1)
orderabstractor_kwargs = dict(embedding_dim=embedding_dim, abstractor_kwargs=abstractor_kwargs)
encoder_kwargs = dict(num_layers=1, num_heads=1, dff=64, dropout_rate=0.1)

for seed in tqdm(seeds):
    X_train_shuffled, y_train_shuffled = shuffle(copy.deepcopy(X_train), copy.deepcopy(y_train), random_state=seed)
    seed_accuracies = []
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    for train_size in range(10, 250, 10):
        X_train_ = copy.deepcopy(X_train_shuffled)[:train_size]
        y_train_ = copy.deepcopy(y_train_shuffled)[:train_size]
        transformer_model = RESOLVEModel(embedding_dim=64)
        transformer_model.compile(loss='binary_crossentropy', optimizer=create_opt(), metrics=['binary_accuracy'])
        transformer_model(X_train_[:32])
        transformer_model.fit(X_train_, y_train_, validation_data=(X_val, y_val), epochs=100, verbose=0, batch_size=128)
        results = transformer_model.evaluate(X_test, y_test, return_dict=True,verbose=0)
        del(transformer_model)
        gc.collect()  # Run garbage collection
        tf.keras.backend.clear_session()  # Clear the Keras session
        seed_accuracies.append(results['binary_accuracy'])
    accuracies.append(seed_accuracies)
accuracies_np = np.array(accuracies)
np.save('final_accuracies_resolve.npy', accuracies_np)


for seed in tqdm(seeds):
    X_train_shuffled, y_train_shuffled = shuffle(copy.deepcopy(X_train), copy.deepcopy(y_train), random_state=seed)
    seed_accuracies = []
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)    
    for train_size in range(10, 250, 10):
        X_train_ = copy.deepcopy(X_train_shuffled)[:train_size]
        y_train_ = copy.deepcopy(y_train_shuffled)[:train_size]
        encoder_kwargs = dict(num_layers=1, num_heads=1, dff=64, dropout_rate=0.1)
        #transformer_model = NVSAOrderModel(embedding_dim=64)
        transformer_model = TransformerOrderModel(embedding_dim=64,encoder_kwargs=encoder_kwargs)
        transformer_model.compile(loss='binary_crossentropy', optimizer=create_opt(), metrics=['binary_accuracy'])
        transformer_model(X_train_[:32])
        transformer_model.fit(X_train_, y_train_, validation_data=(X_val, y_val), epochs=100, verbose=0, batch_size=128)
        results = transformer_model.evaluate(X_test, y_test, return_dict=True,verbose=0)
        del(transformer_model)
        gc.collect()  # Run garbage collection
        tf.keras.backend.clear_session()  # Clear the Keras session
        seed_accuracies.append(results['binary_accuracy'])
    accuracies.append(seed_accuracies)
accuracies_np = np.array(accuracies)
np.save('final_accuracies_transformer.npy', accuracies_np)


for seed in tqdm(seeds):
    X_train_shuffled, y_train_shuffled = shuffle(copy.deepcopy(X_train), copy.deepcopy(y_train), random_state=seed)
    seed_accuracies = []
    for train_size in range(10, 250, 10):
        X_train_ = copy.deepcopy(X_train_shuffled)[:train_size]
        y_train_ = copy.deepcopy(y_train_shuffled)[:train_size]
        encoder_kwargs = dict(num_layers=1, num_heads=1, dff=64, dropout_rate=0.1)
        #transformer_model = NVSAOrderModel(embedding_dim=64)
        transformer_model = MLPModel(**mlp_kwargs, name='mlp')
        transformer_model.compile(loss='binary_crossentropy', optimizer=create_opt(), metrics=['binary_accuracy'])
        transformer_model(X_train_[:32])
        transformer_model.fit(X_train_, y_train_, validation_data=(X_val, y_val), epochs=100, verbose=0, batch_size=128)
        results = transformer_model.evaluate(X_test, y_test, return_dict=True,verbose=0)
        del(transformer_model)
        gc.collect()  # Run garbage collection
        tf.keras.backend.clear_session()  # Clear the Keras session
        seed_accuracies.append(results['binary_accuracy'])
    accuracies.append(seed_accuracies)
accuracies_np = np.array(accuracies)
np.save('final_accuracies_mlp.npy', accuracies_np)



for seed in tqdm(seeds):
    X_train_shuffled, y_train_shuffled = shuffle(copy.deepcopy(X_train), copy.deepcopy(y_train), random_state=seed)
    seed_accuracies = []
    for train_size in range(10, 250, 10):
        X_train_ = copy.deepcopy(X_train_shuffled)[:train_size]
        y_train_ = copy.deepcopy(y_train_shuffled)[:train_size]
        encoder_kwargs = dict(num_layers=1, num_heads=1, dff=64, dropout_rate=0.1)
        transformer_model = AbstractorOrderModel(**orderabstractor_kwargs, name='order_abstractor')
        transformer_model.compile(loss='binary_crossentropy', optimizer=create_opt(), metrics=['binary_accuracy'])
        transformer_model(X_train_[:32])
        transformer_model.fit(X_train_, y_train_, validation_data=(X_val, y_val), epochs=100, verbose=0, batch_size=128)
        results = transformer_model.evaluate(X_test, y_test, return_dict=True,verbose=0)
        del(transformer_model)
        gc.collect()  # Run garbage collection
        tf.keras.backend.clear_session()  # Clear the Keras session
        seed_accuracies.append(results['binary_accuracy'])
    accuracies.append(seed_accuracies)
accuracies_np = np.array(accuracies)
np.save('final_accuracies_abstractor.npy', accuracies_np)


for seed in tqdm(seeds):
    X_train_shuffled, y_train_shuffled = shuffle(copy.deepcopy(X_train), copy.deepcopy(y_train), random_state=seed)
    seed_accuracies = []
    for train_size in range(10, 250, 10):
        X_train_ = copy.deepcopy(X_train_shuffled)[:train_size]
        y_train_ = copy.deepcopy(y_train_shuffled)[:train_size]
        encoder_kwargs = dict(num_layers=1, num_heads=1, dff=64, dropout_rate=0.1)
        transformer_model = build_len_binary(input_shape)
        transformer_model.compile(loss='binary_crossentropy', optimizer=create_opt(), metrics=['binary_accuracy'])
        transformer_model(X_train_[:32])
        transformer_model.fit(X_train_, y_train_, validation_data=(X_val, y_val), epochs=100, verbose=0, batch_size=128)
        results = transformer_model.evaluate(X_test, y_test, return_dict=True,verbose=0)
        del(transformer_model)
        gc.collect()  # Run garbage collection
        tf.keras.backend.clear_session()  # Clear the Keras session
        seed_accuracies.append(results['binary_accuracy'])
    accuracies.append(seed_accuracies)
accuracies_np = np.array(accuracies)
np.save('final_accuracies_len.npy', accuracies_np)

predinet_kwargs = dict(embedding_dim=embedding_dim, predinet_kwargs=dict(key_dim=4, n_heads=4, n_relations=16, add_temp_tag=False))
ordercorelnet_kwargs = dict(embedding_dim=embedding_dim)


for seed in tqdm(seeds):
    X_train_shuffled, y_train_shuffled = shuffle(copy.deepcopy(X_train), copy.deepcopy(y_train), random_state=seed)
    seed_accuracies = []
    for train_size in range(10, 250, 10):
        X_train_ = copy.deepcopy(X_train_shuffled)[:train_size]
        y_train_ = copy.deepcopy(y_train_shuffled)[:train_size]
        encoder_kwargs = dict(num_layers=1, num_heads=1, dff=64, dropout_rate=0.1)
        transformer_model = PrediNetModel(**predinet_kwargs, name='predinet')
        transformer_model.compile(loss='binary_crossentropy', optimizer=create_opt(), metrics=['binary_accuracy'])
        transformer_model(X_train_[:32])
        transformer_model.fit(X_train_, y_train_, validation_data=(X_val, y_val), epochs=100, verbose=0, batch_size=128)
        results = transformer_model.evaluate(X_test, y_test, return_dict=True,verbose=0)
        del(transformer_model)
        gc.collect()  # Run garbage collection
        tf.keras.backend.clear_session()  # Clear the Keras session
        seed_accuracies.append(results['binary_accuracy'])
    accuracies.append(seed_accuracies)
accuracies_np = np.array(accuracies)
np.save('final_accuracies_predinet.npy', accuracies_np)

for seed in tqdm(seeds):
    X_train_shuffled, y_train_shuffled = shuffle(copy.deepcopy(X_train), copy.deepcopy(y_train), random_state=seed)
    seed_accuracies = []
    for train_size in range(10, 250, 10):
        X_train_ = copy.deepcopy(X_train_shuffled)[:train_size]
        y_train_ = copy.deepcopy(y_train_shuffled)[:train_size]
        encoder_kwargs = dict(num_layers=1, num_heads=1, dff=64, dropout_rate=0.1)
        transformer_model = CorelNetModel(**ordercorelnet_kwargs, name='order_corelnet')
        transformer_model.compile(loss='binary_crossentropy', optimizer=create_opt(), metrics=['binary_accuracy'])
        transformer_model(X_train_[:32])
        transformer_model.fit(X_train_, y_train_, validation_data=(X_val, y_val), epochs=100, verbose=0, batch_size=128)
        results = transformer_model.evaluate(X_test, y_test, return_dict=True,verbose=0)
        del(transformer_model)
        gc.collect()  # Run garbage collection
        tf.keras.backend.clear_session()  # Clear the Keras session
        seed_accuracies.append(results['binary_accuracy'])
    accuracies.append(seed_accuracies)
accuracies_np = np.array(accuracies)
np.save('final_accuracies_correlnet.npy', accuracies_np)



 

