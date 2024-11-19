import tensorflow as tf
from tensorflow.keras import layers, Model,Sequential
import time
from transformer_modules import Encoder, Decoder, AddPositionalEmbedding,FeedForward
from abstracters import SymbolicAbstracter, RelationalAbstracter, AblationAbstractor
from tensorflow.keras.regularizers import l2
from multi_head_relation import MultiHeadRelation

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
tf.random.set_seed(2023)



from tensorflow.keras import layers

        
        
                

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff,
            input_vocab, target_vocab, embedding_dim, output_dim,
            dropout_rate=0.1, name='transformer'):
        """A transformer model.

        Args:
            num_layers (int): # of layers in encoder and decoder
            num_heads (int): # of attention heads in attention operations
            dff (int): dimension of feedforward laeyrs
            input_vocab (int or str): if input is tokens, the size of vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            target_vocab (int): if target is tokens, the size of the vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            embedding_dim (int): embedding dimension to use. this is the model dimension.
            output_dim (int): dimension of final output. e.g.: # of classes.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
            name (str, optional): name of model. Defaults to 'transformer'.
        """

        super().__init__(name=name)

        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.encoder = Encoder(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate, name='encoder')

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        #self.decoder = Decoder()
        self.final_layer = layers.Dense(output_dim, name='final_layer')
        self.decoder = Decoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
          dropout_rate=dropout_rate, name='decoder')
        #self.word_to_id = word_to_id           
        #self.end_token = self.word_to_id(';')
        #self.start_token = self.word_to_id('@')    


    def call(self, inputs):
        source, target  = inputs
        x = self.source_embedder(source)
        x = self.pos_embedding_adder_input(x)
        encoder_context = self.encoder(x)
        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)
        x = self.decoder(x=target_embedding, context=encoder_context)
        logits = self.final_layer(x)
        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass
        return logits


          

class Seq2SeqRelationalAbstracter(tf.keras.Model):
    """
    Sequence-to-Sequence Relational Abstracter.
    Uses the architecture X -> Encoder -> RelationalAbstracter -> Decoder -> y.

    Note: 'autoregressive_abstractor.py' implements a more general seq2seq
    abstractor architecture.
    """
    def __init__(self, num_layers, num_heads, dff, rel_attention_activation,
            input_vocab, target_vocab, embedding_dim, output_dim,
            dropout_rate=0.1, name='seq2seq_relational_abstracter'):
        """
        Args:
            num_layers (int): # of layers in encoder and decoder
            num_heads (int): # of attention heads in attention operations
            dff (int): dimension of feedforward layers
            rel_attention_activation (str): the activation function to use in relational attention.
            input_vocab (int or str): if input is tokens, the size of vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            target_vocab (int): if target is tokens, the size of the vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            embedding_dim (int): embedding dimension to use. this is the model dimension.
            output_dim (int): dimension of final output. e.g.: # of classes.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
            name (str, optional): name of model. Defaults to 'seq2seq_relational_abstracter'.
        """

        super().__init__(name=name)

        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        self.encoder = Encoder(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate, name='encoder')
        self.abstracter = RelationalAbstracter(num_layers=num_layers, num_heads=num_heads, dff=dff,
            mha_activation_type=rel_attention_activation, dropout_rate=dropout_rate, name='abstracter')
        self.decoder = Decoder(num_layers=1, num_heads=2, dff=dff,
          dropout_rate=dropout_rate, name='decoder')
        self.final_layer = layers.Dense(output_dim, name='final_layer')
   
    def call(self, inputs):
        source, target  = inputs
        x = self.source_embedder(source)
        x = self.pos_embedding_adder_input(x)
        x = self.encoder(x)
        abstracted_context = self.abstracter(x)
        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)
        x = self.decoder(x=target_embedding, context=abstracted_context)
        logits = self.final_layer(x)
        try:
          del logits._keras_mask
        except AttributeError:
          pass
        return logits
              


class Seq2SeqSymbolicAbstracter(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff, rel_attention_activation,
            input_vocab, target_vocab, embedding_dim, output_dim,
            dropout_rate=0.1, name='seq2seq_symbolic_abstracter'):
        """
        Sequence-to-Sequence Symbolic Abstracter.

        Args:
            num_layers (int): # of layers in encoder and decoder
            num_heads (int): # of attention heads in attention operations
            dff (int): dimension of feedforward layers
            rel_attention_activation (str): the activation function to use in relational attention.
            input_vocab (int or str): if input is tokens, the size of vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            target_vocab (int): if target is tokens, the size of the vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            embedding_dim (int): embedding dimension to use. this is the model dimension.
            output_dim (int): dimension of final output. e.g.: # of classes.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
            name (str, optional): name of model. Defaults to 'seq2seq_symbolic_abstracter'.
        """

        super().__init__(name=name)

        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        self.encoder = Encoder(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate, name='encoder')
        self.abstracter = SymbolicAbstracter(num_layers=num_layers, num_heads=num_heads, dff=dff,
            mha_activation_type=rel_attention_activation, dropout_rate=dropout_rate, name='abstracter')
        self.decoder = Decoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
          dropout_rate=dropout_rate, name='decoder')
        self.final_layer = layers.Dense(output_dim, name='final_layer')


    def call(self, inputs):
        source, target  = inputs

        x = self.source_embedder(source)
        x = self.pos_embedding_adder_input(x)

        encoder_context = self.encoder(x)

        abstracted_context = self.abstracter(encoder_context)

        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)

        x = self.decoder(x=target_embedding, context=abstracted_context)

        logits = self.final_layer(x)

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        return logits


class AutoregressiveAblationAbstractor(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff, mha_activation_type,
            input_vocab, target_vocab, embedding_dim, output_dim,
            use_encoder, use_self_attn,
            dropout_rate=0.1, name='seq2seq_ablation_abstractor'):
        """
        Sequence-to-Sequence Ablation Abstracter.

        A Seq2Seq Abstractor model where the abstractor's cross-attention
        scheme is standard cross-attention rather than relation cross-attention.
        Used to isolate for the effect of relational cross-attention in abstractor models.

        Args:
            num_layers (int): # of layers in encoder and decoder
            num_heads (int): # of attention heads in attention operations
            dff (int): dimension of feedforward layers
            mha_activation_type (str): the activation function to use in AblationAbstractor's cross-attention.
            input_vocab (int or str): if input is tokens, the size of vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            target_vocab (int): if target is tokens, the size of the vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            embedding_dim (int): embedding dimension to use. this is the model dimension.
            output_dim (int): dimension of final output. e.g.: # of classes.
            use_encoder (bool): whether to use a (Transformer) Encoder as first step of processing.
            use_self_attn (bool): whether to use self-attention in AblationAbstractor.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
            name (str, optional): name of model. Defaults to 'seq2seq_relational_abstracter'.
        """

        super().__init__(name=name)

        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        self.use_encoder = use_encoder
        self.use_self_attn = use_self_attn
        if self.use_encoder:
            self.encoder = Encoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
            dropout_rate=dropout_rate, name='encoder')
        self.abstractor = AblationAbstractor(num_layers=num_layers, num_heads=num_heads, dff=dff,
            mha_activation_type=mha_activation_type, use_self_attn=use_self_attn, dropout_rate=dropout_rate,
            name='ablation_abstractor')
        self.decoder = Decoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
          dropout_rate=dropout_rate, name='decoder')
        self.final_layer = layers.Dense(output_dim, name='final_layer')


    def call(self, inputs):
        source, target  = inputs

        x = self.source_embedder(source)
        x = self.pos_embedding_adder_input(x)

        if self.use_encoder:
            encoder_context = self.encoder(x)
        else:
            encoder_context = x

        abstracted_context = self.abstractor(encoder_context)
        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)

        x = self.decoder(x=target_embedding, context=abstracted_context)

        logits = self.final_layer(x)

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        return logits
        





class Seq2SeqCorelNet(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff, rel_attention_activation,
            input_vocab, target_vocab, embedding_dim, output_dim,encoder_kwargs,
            dropout_rate=0.1, name='seq2seq_relational_abstracter'):
        """
        Sequence-to-Sequence Ablation Abstracter.

        A Seq2Seq Abstractor model where the abstractor's cross-attention
        scheme is standard cross-attention rather than relation cross-attention.
        Used to isolate for the effect of relational cross-attention in abstractor models.

        Args:
            num_layers (int): # of layers in encoder and decoder
            num_heads (int): # of attention heads in attention operations
            dff (int): dimension of feedforward layers
            mha_activation_type (str): the activation function to use in AblationAbstractor's cross-attention.
            input_vocab (int or str): if input is tokens, the size of vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            target_vocab (int): if target is tokens, the size of the vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            embedding_dim (int): embedding dimension to use. this is the model dimension.
            output_dim (int): dimension of final output. e.g.: # of classes.
            use_encoder (bool): whether to use a (Transformer) Encoder as first step of processing.
            use_self_attn (bool): whether to use self-attention in AblationAbstractor.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
            name (str, optional): name of model. Defaults to 'seq2seq_relational_abstracter'.
        """

        super().__init__(name=name)

        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        self.encoder = Encoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
            dropout_rate=dropout_rate, name='encoder')
            
        self.decoder = Decoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
          dropout_rate=dropout_rate, name='decoder')
        self.final_layer = layers.Dense(output_dim, name='final_layer')
        self.mhr = MultiHeadRelation(rel_dim=64, proj_dim=None, symmetric=True, dense_kwargs=dict(use_bias=False))


    def call(self, inputs):
        source, target  = inputs

        x = self.source_embedder(source)
        x = self.pos_embedding_adder_input(x)
        x = self.encoder(x)
        abstracted_context = self.mhr(x)
        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)
        x = self.decoder(x=target_embedding, context=abstracted_context)

        logits = self.final_layer(x)

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        return logits
                



class BipolarDense(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(BipolarDense, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros",
                                 trainable=True)

    def call(self, inputs, training=False):
        weights = tf.math.sign(self.w)
        output = tf.matmul(inputs, weights) + self.b
        return output
    
class AddPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_length=1024, name="add_positional_embedding"):
        super().__init__(name=name)
        self.max_length = max_length

    def build(self, input_shape):
        _, self.seq_length, self.vec_dim = input_shape
        self.max_length = max(self.max_length, self.seq_length)
        self.pos_encoding = create_positional_encoding(length=self.max_length, depth=self.vec_dim)

    def call(self, x):
        length = tf.shape(x)[1]

        # This factor sets the relative scale of the embedding and positional_encoding.
        x *= tf.math.sqrt(tf.cast(self.vec_dim, tf.float32))

        # add positional encoding
        x = x + self.pos_encoding[tf.newaxis, :length, :]

        return x



class AnotherSmallConvNet(tf.keras.Model):
    def __init__(self):
        super(AnotherSmallConvNet, self).__init__()
        self.conv1 = BipolarDense(1024)
        self.dropout = layers.Dropout(0.3)
        self.ln = layers.BatchNormalization()
    def call(self, x):
        x = self.conv1(x)  
        x = self.ln(x)
        x = tf.nn.silu(x)
        x = self.dropout(x, training=True)
        return x
        
class ConvLayerWithLearnableH(tf.keras.layers.Layer):
    def __init__(self, S, h_size):
        super(ConvLayerWithLearnableH, self).__init__()
        # Initialize h as a trainable variable
        self.h = self.add_weight(shape=(S,h_size), initializer='random_normal', trainable=True)
        self.h_s = h_size
    def call(self, x_batch):
        S = tf.shape(x_batch)[1]
        def convolve_step(i):
            # Flip the impulse response h[n] for proper convolution
            #h_flipped = tf.reverse(self.h[i,:], axis=[0])
            h_flipped = self.h[i,:]
            # Pad each signal in the batch to allow for full convolution
            padded_x_batch = tf.pad(x_batch[:, i, :], [[0, 0], [self.h_s-1, self.h_s-1]])

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
        self.conv1 = ConvLayerWithLearnableH(5,1024-64+1)
        self.dropout = layers.Dropout(0.3)
        self.ln1 = layers.LayerNormalization()

    def call(self, x):
        x = self.conv1(x)
        x = tf.nn.silu(x)
        x = self.ln1(x)
        x = self.dropout(x, training=True)
        return x

        
        
class HDSymbolicAttention(layers.Layer):
    def __init__(self, VSA_dim, seq_N, embedding_dim, symbolic, name):
        super(HDSymbolicAttention, self).__init__(name=name)
        self.d_model = VSA_dim
        self.n_seq = seq_N
        self.embd_size = embedding_dim
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
        self.Symbols = tf.Variable(normal_initializer(shape=(input_shape[1], self.embd_size)), name='symbols', trainable=True)
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
        O = tf.nn.silu(attention_output*symbol_projected)
        #V = attention_output
        return O

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.d_model)


        
class Binding(tf.keras.Model):
    def __init__(self,
                 mask_token,
                 encoder_kwargs,
                 input_vocab,
                 target_vocab,
                 embedding_dim,
                 VSA_dim,
                 scale_source,
                 scale_target,
                 name="Binding"):

        super().__init__(name=name)
        
        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")
   
        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')  
        self.bn_t = layers.BatchNormalization(name='batch_norm1') 
        self.dropout = layers.Dropout(0.4, name='dropout0') 
        self.encoder = Encoder(**encoder_kwargs, name='encoder')

    def call(self, inputs):
        source, target = inputs
        source = self.source_embedder(source)
        source = self.pos_embedding_adder_input(source)
        source = self.encoder(source)
        
        target_hd = self.target_embedder(target)
        target_hd = self.pos_embedding_adder_target(target_hd)
        target_hd = self.dropout(target_hd, training=True)
        return source, target_hd

class Seq2SeqLARS_VSA(tf.keras.Model):
    def __init__(self, num_layers, num_heads, num_heads_H, dff, rel_attention_activation,
            input_vocab, target_vocab, embedding_dim, output_dim, VSA_dim, seq_N, symbolic, encoder_kwargs, 
             dropout_rate=0.1, name='seq2seq_relational_abstracter'):
        super().__init__(name=name)
        self.mha = HDSymbolicAttention(VSA_dim, seq_N, embedding_dim, symbolic, name='multi_head_attention') 
        self.final_layer = layers.Dense(target_vocab, name='final_layer') 
        self.bn1 = layers.LayerNormalization(epsilon=1e-5,name='batch_norm1') 
        self.dropout = layers.Dropout(0.2)
        self.add = layers.Add(name='add_layer')
        self.bnd = Binding(0, encoder_kwargs, input_vocab, target_vocab, embedding_dim, VSA_dim, 1, 1, name='binding')
        self.down = layers.Dense(64, activation='relu')
        self.down2 = layers.Dense(64,activation='relu')
        self.decoder = Decoder(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate, name='decoder')

    def encoder(self, inputs):
        source, h_t= self.bnd(inputs)    
        h_a = self.mha(source)  # Abstract_output 
        return h_t, h_a

    def call(self, inputs):
        h_t, h_a = self.encoder(inputs)
        h_a_d = self.down(h_a)
        h_t_d = self.down(h_t)
        h_a_d = self.bn1(h_a_d,training=True)
        h_a_d = self.dropout(h_a_d)
        x = self.decoder(x=h_t_d, context=h_a_d)        
        logits = self.final_layer(x)
        return logits                                  

        
        
        
        
        
                
        


