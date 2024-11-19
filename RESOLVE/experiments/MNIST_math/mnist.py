import tensorflow as tf
import numpy as np
#-------------------------------------------------------------------
#    PLEASE MOVE THIS FILE TO THE MAIN FOLDER TO RUN
#

# Load MNIST dataset
(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the images
mnist_train_images = mnist_train_images.reshape(-1,784)/ 255.0
mnist_test_images = mnist_test_images.reshape(-1,784)/ 255.0

# Function to create the dataset
def create_custom_dataset(images, labels, num_samples):
    idx = np.random.randint(0, len(images), size=(num_samples, 2))  # Randomly select two images
    pairs_images = []
    targets = []
    
    for i in range(num_samples):
        img1 = images[idx[i, 0]]
        img2 = images[idx[i, 1]]
        
        label1 = labels[idx[i, 0]]
        label2 = labels[idx[i, 1]]
        
        # Concatenate the two images into one
        combined_image = np.stack([img1, img2], axis=1)  # Stack images along the last axis
        
        # Calculate the output as abs(2*m - n)
        target = abs(3*label1 - 2*label2)
        
        pairs_images.append(combined_image)
        targets.append(target)    
    pairs_images = np.array(pairs_images)
    targets = np.array(targets)
    
    return pairs_images, targets

# Create train, validation, and test datasets
train_images, y_train = create_custom_dataset(mnist_train_images, mnist_train_labels, 10000)
val_images, y_val = create_custom_dataset(mnist_train_images, mnist_train_labels, 500)
test_images, y_test = create_custom_dataset(mnist_test_images, mnist_test_labels, 2000)


X_train = train_images.reshape((-1,2,784))
X_val = val_images.reshape((-1,2,784))
X_test = test_images.reshape((-1,2,784))
object_seqs_train = X_train

from transformer_modules import Encoder, Decoder, AddPositionalEmbedding
from abstracters import SymbolicAbstracter, RelationalAbstracter









create_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='binary_crossentropy')
create_opt = lambda : tf.keras.optimizers.Adam()

def create_callbacks():
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', restore_best_weights=True, patience=50, start_from_epoch=30)]
    return callbacks








import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tqdm import tqdm

class PrediNet(tf.keras.layers.Layer):
    """PrediNet layer (Shanahan et al. 2020)"""

    def __init__(self, key_dim, n_heads, n_relations, add_temp_tag=False):
        """create PrediNet layer.

        Parameters
        ----------
        key_dim : int
            key dimension
        n_heads : int
            number of heads
        n_relations : int
            number of relations
        add_temp_tag : bool, optional
            whether to add temporal tag to object representations, by default False
        """

        super(PrediNet, self).__init__()
        self.key_dim = key_dim
        self.n_heads = n_heads
        self.n_relations = n_relations
        self.add_temp_tag = add_temp_tag

    def build(self, input_shape):
        _, self.n_objs, obj_dim = input_shape

        self.obj_dim = obj_dim
        self.obj_tagged_dim = self.obj_dim + 1

        self.W_k = layers.Dense(self.key_dim, use_bias=False)
        self.W_q1 = layers.Dense(self.n_heads * self.key_dim, use_bias=False)
        self.W_q2 = layers.Dense(self.n_heads * self.key_dim, use_bias=False)
        self.W_s = layers.Dense(self.n_relations, use_bias=False)

        self.relu = layers.ReLU()
        self.softmax = layers.Softmax(axis=1)
        self.flatten = layers.Flatten()

        # create temporal tag
        if self.add_temp_tag:
            self.temp_tag = tf.convert_to_tensor(np.arange(self.n_objs), dtype=tf.float32)
            self.temp_tag = tf.expand_dims(self.temp_tag, axis=0)
            self.temp_tag = tf.expand_dims(self.temp_tag, axis=2)


    def call(self, obj_seq):

        # append temporal tag to all z
        if self.add_temp_tag:
            temp_tag = tf.tile(self.temp_tag, multiples=[tf.shape(obj_seq)[0], 1, 1])
            obj_seq = tf.concat([obj_seq, temp_tag], axis=2)

        # Get keys for all objects in sequence
        K = self.W_k(obj_seq)

        # get queries for objects 1 and 2
        obj_seq_flat = self.flatten(obj_seq)
        Q1 = self.W_q1(obj_seq_flat)
        Q2 = self.W_q2(obj_seq_flat)

        # reshape queries to separate heads
        Q1_reshaped = tf.reshape(Q1, shape=(-1, self.n_heads, self.key_dim))
        Q2_reshaped = tf.reshape(Q2, shape=(-1, self.n_heads, self.key_dim))

        # extract attended objects
        E1 = (self.softmax(tf.reduce_sum(Q1_reshaped[:, tf.newaxis, :, :] * K[:, :, tf.newaxis, :], axis=3))
             [:, :, :, tf.newaxis] * obj_seq[:, :, tf.newaxis, :])
        E1 = tf.reduce_sum(E1, axis=1)
        E2 = (self.softmax(tf.reduce_sum(Q2_reshaped[:, tf.newaxis, :, :] * K[:, :, tf.newaxis, :], axis=3))
              [:, :, :, tf.newaxis] * obj_seq[:, :, tf.newaxis, :])
        E2 = tf.reduce_sum(E2, axis=1)

        # compute relation vector
        D = self.W_s(E1) - self.W_s(E2)

        # add temporal position tag
        if self.add_temp_tag:
            D = tf.concat([D, E1[:, :, -1][:, :, tf.newaxis], E2[:, :, -1][:, :, tf.newaxis]], axis=2)

        R = self.flatten(D) # concatenate heads

        return R


def create_predinet(encoder_kwargs, embedding_dim, dropout_rate=0.1, name='predinet'):
    inputs = layers.Input(shape=object_seqs_train.shape[1:], name='input_seq')
    object_embedder = tf.keras.Sequential([layers.Dense(embedding_dim)])
    source_embedder = layers.TimeDistributed(object_embedder, name='source_embedder')
    pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
    #if encoder_kwargs:
    encoder = Encoder(**encoder_kwargs, name='encoder')
    predinet = PrediNet(key_dim=64, n_heads=2 ,n_relations= 4, add_temp_tag=True)
    flattener = layers.Flatten()
    final_layer = layers.Dense(28, name='final_layer')
    
    x = source_embedder(inputs)
    x = pos_embedding_adder_input(x)
    x = predinet(x)
    x = flattener(x)
    logits = final_layer(x)

    abstractor_model = tf.keras.Model(inputs=inputs, outputs=logits, name=name)
    return abstractor_model
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
accuracies = []
seeds = [2020, 2021, 2022]


class ConvLayerWithLearnableH(tf.keras.layers.Layer):
    def __init__(self, h_size):
        super(ConvLayerWithLearnableH, self).__init__()
        # Initialize h as a trainable variable
        self.h_size = h_size
    def build(self, input_shape):
        self.h = self.add_weight(shape=(2, self.h_size), initializer='random_normal', trainable=True)
    def call(self, x_batch):
        S = 2

        def convolve_step(i):
            # Flip the impulse response h[n] for proper convolution
            h_flipped = self.h[i,:]

            # Pad each signal in the batch to allow for full convolution
            padded_x_batch = tf.pad(x_batch[:, i, :], [[0, 0], [960, 960]])

            # Reshape both signals for 1D convolution
            x_batch_reshaped = tf.reshape(padded_x_batch, [tf.shape(padded_x_batch)[0], -1, 1])  # Shape: (batch_size, signal_length, 1)
            h_reshaped = tf.reshape(h_flipped, [-1, 1, 1])  # Reshape filter (kernel) after flipping

            # Perform 1D convolution using VALID padding for each signal in the batch
            y_batch = tf.nn.conv1d(x_batch_reshaped, h_reshaped, stride=1, padding='VALID')

            # Reshape the result back to 2D (batch_size, output_length)
            return tf.squeeze(y_batch)

        # Use tf.map_fn to apply the convolution step over the sequence dimension
        output = tf.map_fn(convolve_step, tf.range(S), dtype=tf.float32)

        # Transpose the output tensor to ensure the correct shape (batch_size, S, output_length)
        output = tf.transpose(output, perm=[1, 0, 2])

        return output


class SmallConvNet(tf.keras.Model):
    def __init__(self):
        super(SmallConvNet, self).__init__()
        #self.conv1 = layers.Conv1D(filters=128, kernel_size=5, padding='same')
        self.encoder = ConvLayerWithLearnableH( h_size=1024-64+1)
        self.dropout = layers.Dropout(0.3)
        self.ln1 = layers.LayerNormalization()
        #self.ln2 = layers.BatchNormalization()

    def call(self, x):
        #x = self.conv1(x)
        #x = tf.nn.silu(x)
        #x = self.ln2(x)
        x = self.encoder(x) 
        x = self.ln1(x)
        x = tf.nn.silu(x)    
        x = self.dropout(x, training=True)
        return x


class HDSymbolicAttention(layers.Layer):
    def __init__(self, d_model, embd_size, training, symbolic, name="hd_symbolic_attention", **kwargs):
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
        #V = tf.einsum('bij,kj->bkj',values,self.www)
        #S = tf.einsum('bij,kj->bkj',self.S3 +self.Symbols,self.www)
        values_projected = self.process(values)
        symbol_projected = self.process(self.S3 +self.Symbols)   

        #values_projected = tf.nn.tanh(tf.einsum('bij,jk->bik',values,hd_encoder))
        if self.symbolic:
            scores = self.create_cosine_similarity_matrix(values_projected,symbol_projected)
        else:
            scores = self.create_cosine_similarity_matrix(values_projected,values_projected) 
        attention_output = tf.matmul(scores,values_projected)
        O = tf.nn.swish(attention_output)*symbol_projected
        return O

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.d_model)


from transformer_modules import Encoder, Decoder, AddPositionalEmbedding
from abstracters import SymbolicAbstracter, RelationalAbstracter

class RESOLVE(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff, embedding_dim, training, symbolic, dropout_rate=0.1):
        super(RESOLVE, self).__init__()
        
        self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.encoder = Encoder(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate, name='encoder')
        self.mha1 = HDSymbolicAttention(1024, 64,training = training, symbolic = symbolic)
        self.flattener = layers.Flatten()
        self.dropout = layers.Dropout(0.3)
        self.bn2 = layers.LayerNormalization()
        self.final_layer = layers.Dense(28, name='final_layer')

    def call(self, inputs):
        x = self.source_embedder(inputs)
        x = self.pos_embedding_adder_input(x)
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.mha1(x) 
        x = self.bn2(x)
        x = self.flattener(x)
        x = self.final_layer(x)
        return x
    

import numpy as np
from tqdm import tqdm

from multi_head_relation import MultiHeadRelation

def create_corelnet(embedding_dim, activation='softmax', name='corelnet'):
    inputs = layers.Input(shape=object_seqs_train.shape[1:], name='input_seq')
    object_embedder = tf.keras.Sequential([layers.Dense(embedding_dim)])
    source_embedder = layers.TimeDistributed(object_embedder, name='source_embedder')
    activation = layers.Softmax(axis=1) if activation == 'softmax' else layers.Activation(activation)
    mhr = MultiHeadRelation(rel_dim=1, proj_dim=None, symmetric=True, dense_kwargs=dict(use_bias=False))
    flattener = layers.Flatten()
    final_layer = layers.Dense(28, name='final_layer')

    x = source_embedder(inputs)
    x = mhr(x)
    x = activation(x)
    x = flattener(x)
    logits = final_layer(x)

    corelnet_model = tf.keras.Model(inputs=inputs, outputs=logits, name=name)
    return corelnet_model


corelnet_model_kwargs = dict(embedding_dim=64, activation='softmax')
corelnet_model_nosoftmax_kwargs = dict(embedding_dim=64, activation='linear')
corelnet_model = create_corelnet(**corelnet_model_nosoftmax_kwargs)
corelnet_model.compile(loss=create_loss, optimizer=create_opt(), metrics=['acc'])
corelnet_model(X_train[:32]); # build
corelnet_model.summary()


import numpy as np
from sklearn.utils import shuffle
accuracies = []
seeds = [2020, 2021, 2022]
for seed in tqdm(seeds):
    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=seed)
    seed_accuracies = []
    for train_size in range(1000, 10000, 1000):
        X_train_ = X_train_shuffled[:train_size]
        y_train_ = y_train_shuffled[:train_size]
        corelnet_model_kwargs = dict(embedding_dim=64, activation='softmax')
        corelnet_model_nosoftmax_kwargs = dict(embedding_dim=64, activation='linear')
        transformer_model = create_corelnet(**corelnet_model_nosoftmax_kwargs)
        transformer_model.compile(loss=create_loss, optimizer=create_opt(), metrics=['acc'])
        transformer_model.fit(X_train_, y_train_, validation_data=(X_val, y_val), epochs=100, verbose=0, batch_size=512, callbacks=create_callbacks())
        results = transformer_model.evaluate(X_test, y_test, return_dict=True)
        seed_accuracies.append(results['acc'])
    accuracies.append(seed_accuracies)
accuracies_np = np.array(accuracies)
np.save('accuracies_correlnet.npy', accuracies_np)
mean_accuracy = np.mean(accuracies_np)
print(f"Mean accuracy over all seeds: {mean_accuracy}")

from sklearn.utils import shuffle
accuracies = []
seeds = [2020, 2021, 2022]
for seed in tqdm(seeds):
    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=seed)
    seed_accuracies = []
    for train_size in range(1000, 10000, 1000):
        X_train_ = X_train_shuffled[:train_size]
        y_train_ = y_train_shuffled[:train_size]
        transformer_model = RESOLVE(num_layers=1, num_heads=2, dff=64, embedding_dim=64, dropout_rate=0.1, training=True, symbolic=False)
        transformer_model(X_train[:32])
        transformer_model.compile(loss=create_loss, optimizer=create_opt(), metrics=['acc'])
        print(X_train_.shape)
        transformer_model.fit(X_train_, y_train_, validation_data=(X_val, y_val), epochs=100, verbose=0, batch_size=512, callbacks=create_callbacks())
        results = transformer_model.evaluate(X_test, y_test, return_dict=True)
        seed_accuracies.append(results['acc'])
    accuracies.append(seed_accuracies)
accuracies_np = np.array(accuracies)
np.save('./accuracies.npy', accuracies_np)
mean_accuracy = np.mean(accuracies_np)
print(f"Mean accuracy over all seeds: {mean_accuracy}")





from transformer_modules import Encoder, Decoder, AddPositionalEmbedding
from abstracters import RelationalAbstracter
from abstractor import Abstractor

def create_abstractor(encoder_kwargs, abstractor_kwargs, embedding_dim, dropout_rate=0.1, name='abstractor'):
    inputs = layers.Input(shape=object_seqs_train.shape[1:], name='input_seq')
    object_embedder = tf.keras.Sequential([layers.Dense(embedding_dim)])
    source_embedder = layers.TimeDistributed(object_embedder, name='source_embedder')
    pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
    #if encoder_kwargs:
    encoder = Encoder(**encoder_kwargs, name='encoder')
    abstractor = RelationalAbstracter(**abstractor_kwargs, name='abstractor')
    flattener = layers.Flatten()
    final_layer = layers.Dense(28, name='final_layer')
    fc = layers.Dense(64,activation='relu')
    
    x = source_embedder(inputs)
    x = pos_embedding_adder_input(x)
    x = encoder(x)   
    x = abstractor(x)
    x = flattener(x)
    logits = final_layer(x)

    abstractor_model = tf.keras.Model(inputs=inputs, outputs=logits, name=name)
    return abstractor_model


encoder_kwargs = dict(num_layers=1, num_heads=2, dff=64, dropout_rate=0.1)
abstractor_kwargs = dict(num_layers=1, num_heads=2, dff=64,
     use_pos_embedding=True, mha_activation_type='softmax', dropout_rate=0.1)

abstractor_model_kwargs = dict(encoder_kwargs=encoder_kwargs, abstractor_kwargs=abstractor_kwargs, embedding_dim=64)
abstractor_model = create_abstractor(**abstractor_model_kwargs)

abstractor_model.compile(loss=create_loss, optimizer=create_opt(), metrics=['acc'])
abstractor_model(X_train[:32]); # build
abstractor_model.summary()


import numpy as np
from sklearn.utils import shuffle
accuracies = []
seeds = [2020, 2021, 2022]
for seed in tqdm(seeds):
    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=seed)
    seed_accuracies = []
    for train_size in range(1000, 10000, 1000):
        X_train_ = X_train_shuffled[:train_size]
        y_train_ = y_train_shuffled[:train_size]
        transformer_model = create_abstractor(**abstractor_model_kwargs)
        transformer_model.compile(loss=create_loss, optimizer=create_opt(), metrics=['acc'])
        transformer_model.fit(X_train_, y_train_, validation_data=(X_val, y_val), epochs=100, verbose=0, batch_size=512, callbacks=create_callbacks())
        results = transformer_model.evaluate(X_test, y_test, return_dict=True)
        seed_accuracies.append(results['acc'])
    accuracies.append(seed_accuracies)
accuracies_np = np.array(accuracies)
np.save('./accuracies_abstractor.npy', accuracies_np)
mean_accuracy = np.mean(accuracies_np)
print(f"Mean accuracy over all seeds: {mean_accuracy}")




from transformer_modules import Encoder, Decoder, AddPositionalEmbedding
from abstracters import RelationalAbstracter
from abstractor import Abstractor

def create_transformer(encoder_kwargs, embedding_dim, dropout_rate=0.1, name='abstractor'):
    inputs = layers.Input(shape=object_seqs_train.shape[1:], name='input_seq')
    object_embedder = tf.keras.Sequential([layers.Dense(embedding_dim)])
    source_embedder = layers.TimeDistributed(object_embedder, name='source_embedder')
    pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
    encoder = Encoder(**encoder_kwargs, name='encoder0')
    flattener = layers.Flatten()
    final_layer = layers.Dense(28, name='final_layer')
    #fc = layers.Dense(64,activation='relu')
    

    x = source_embedder(inputs)
    x = pos_embedding_adder_input(x)
    x = encoder(x)
    #x = fc(x)   
    x = flattener(x)
    logits = final_layer(x)

    abstractor_model = tf.keras.Model(inputs=inputs, outputs=logits, name=name)
    return abstractor_model



encoder_kwargs = dict(num_layers=2, num_heads=2, dff=64, dropout_rate=0.05)
import numpy as np
from sklearn.utils import shuffle
accuracies = []
seeds = [2020, 2021, 2022]
for seed in tqdm(seeds):
    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=seed)
    seed_accuracies = []
    for train_size in range(1000, 10000, 1000):
        X_train_ = X_train_shuffled[:train_size]
        y_train_ = y_train_shuffled[:train_size]
        transformer_model = create_transformer(encoder_kwargs, embedding_dim=64)
        transformer_model.compile(loss=create_loss, optimizer=create_opt(), metrics=['acc'])
        transformer_model.fit(X_train_, y_train_, validation_data=(X_val, y_val), epochs=100, verbose=0, batch_size=512)
        results = transformer_model.evaluate(X_test, y_test, return_dict=True)
        seed_accuracies.append(results['acc'])
    accuracies.append(seed_accuracies)
accuracies_np = np.array(accuracies)
np.save('./accuracies_transformer.npy', accuracies_np)
mean_accuracy = np.mean(accuracies_np)
print(f"Mean accuracy over all seeds: {mean_accuracy}")




