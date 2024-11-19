import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import layers
from transformer_modules import Encoder, AddPositionalEmbedding
from multi_attention_decoder import MultiAttentionDecoder
from abstractor import Abstractor
from abstracters import RelationalAbstracter, SymbolicAbstracter
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras import Model
import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal




class AutoregressiveAbstractor(tf.keras.Model):
    """
    An implementation of an Abstractor-based Transformer module.

    This supports several architectures, including:
    a) X -> Abstractor -> Decoder -> Y
    b) X -> Encoder -> Abstractor -> Decoder -> Y
    c) X -> [Encoder, Abstractor] -> Decoder -> Y
    d) X -> Encoder -> Abstractor; [Encoder, Abstractor] -> Decoder -> Y
    """
    def __init__(self,
            encoder_kwargs,
            abstractor_kwargs,
            decoder_kwargs,
            input_vocab,
            target_vocab,
            embedding_dim,
            output_dim,
            word_to_id,
            abstractor_type='relational', # 'abstractor', 'simple', 'relational', or 'symbolic'
            abstractor_on='input', # 'input' or 'encoder'
            decoder_on='abstractor', # 'abstractor' or 'encoder-abstractor'
            name=None):
        """Creates an autoregressive Abstractor model.

        Parameters
        ----------
        encoder_kwargs : dict
            kwargs for the Encoder module. Can be set to None if architecture does not use an encoder.
        abstractor_kwargs : dict
            kwargs for the Abstractor model. Should match `abstractor_type`
        decoder_kwargs : dict
            kwargs for Decoder module.
        input_vocab : int or 'vector'
            if input is tokens, the size of vocabulary as an int.
            if input is vectors, the string 'vector'. used to create embedder.
        target_vocab : int or 'vector'
            if input is tokens, the size of vocabulary as an int.
            if input is vectors, the string 'vector'. used to create embedder.
        embedding_dim : int or tuple[int]
            dimension of embedding (input will be transformed to this dimension).
        output_dim : int
            dimension of final output. e.g.: # of classes.
        abstractor_type : 'abstractor', 'relational', or 'symbolic', optional
            The type of Abstractor to use, by default 'relational'
        abstractor_on: 'input' or 'encoder'
            what the abstractor should take as input.
        decoder_on: 'abstractor' or 'encoder-abstractor'
            what should form the decoder's 'context'.
            if 'abstractor' the context is the output of the abstractor.
            if 'encoder-abstractor' the context is the concatenation of the outputs of the encoder and decoder. 
        """

        super().__init__(name=name)

        # set params
        self.relation_on = abstractor_on
        self.decoder_on = decoder_on
        self.abstractor_type = abstractor_type
        self.word_to_id = word_to_id           
        self.end_token = self.word_to_id(';')
        self.start_token = self.word_to_id('@')  
        # if relation is computed on inputs and the decoder attends only to the abstractor,
        # there is no need for an encoder
        if (abstractor_on, decoder_on) == ('input', 'abstractor'):
            self.use_encoder = False
            print(f'NOTE: no encoder will be used since relation_on={abstractor_on} and decoder_on={decoder_on}')
        else:
            self.use_encoder = True

        # set up source and target embedders
        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.Dense(embedding_dim, name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.Dense(embedding_dim, name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        # initialize layers
        if self.use_encoder:
            self.encoder = Encoder(**encoder_kwargs, name='encoder')

        # initialize the abstractor based on requested type
        if abstractor_type == 'abstractor':
            self.abstractor = Abstractor(**abstractor_kwargs, name='abstractor')
        elif abstractor_type == 'relational':
            self.abstractor = RelationalAbstracter(**abstractor_kwargs, name='abstractor')
        elif abstractor_type == 'symbolic':
            self.abstractor = SymbolicAbstracter(**abstractor_kwargs, name='abstractor')
        else:
            raise ValueError(f'unexpected `abstracter_type` argument {abstractor_type}')

        # initialize decoder
        self.decoder = MultiAttentionDecoder(**decoder_kwargs, name='decoder')

        # initialize final prediction layer
        self.final_layer = layers.Dense(output_dim, name='final_layer')


    def call(self, inputs):
        source, target = inputs # get source and target from inputs

        # embed source and add positional embedding
        source = self.source_embedder(source)
        source = self.pos_embedding_adder_input(source)

        # pass input to Encoder
        if self.use_encoder:
            encoder_context = self.encoder(source)

        # compute abstracted context (either directly on embedded input or on encoder output)
        if self.relation_on == 'input':
            abstracted_context = self.abstractor(source)
        elif self.relation_on == 'encoder':
            abstracted_context = self.abstractor(encoder_context)
        else:
            raise ValueError()

        # embed target and add positional embedding
        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)

        # decode context (either abstractor only or concatenation of encoder and abstractor outputs)
        if self.decoder_on == 'abstractor':
            decoder_inputs = [target_embedding, abstracted_context]
        elif self.decoder_on == 'encoder-abstractor':
            decoder_inputs = [target_embedding, encoder_context, abstracted_context]
        else:
            raise ValueError()

        x = self.decoder(decoder_inputs)

        # produce final prediction
        logits = self.final_layer(x)
        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics. b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        return logits
    def get_initial_state(self, context, max_length):
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        zeros_tensor = tf.zeros([batch_size, max_length - 1], dtype=tf.int64)
        start_target = tf.concat([start_tokens, zeros_tensor], axis=1)
        return start_target, done

    def get_next_token(self, context, target, iteration, done, temperature=0.0):
        logits = self((context, target))
        pred = tf.argmax(logits, axis=-1)
        done = done | (tf.expand_dims(pred[:, iteration], axis=-1) == self.end_token)
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), tf.expand_dims(pred[:, iteration], axis=-1))
        return next_token, done

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.context_text_processor(texts)
        return context

    def translate(self, context, max_length=30, temperature=0.0):
        batch_size = tf.shape(context)[0]
        tokens = []
        next_target, done = self.get_initial_state(context, max_length)
        
        for iteration in range(max_length):
            next_token, done = self.get_next_token(context, next_target, iteration, done)
            zeros_tokens = tf.zeros([batch_size, max_length - iteration - 1], dtype=tf.int64)
            next_target = tf.concat([next_target[:, :iteration + 1], next_token], axis=1)
            next_target = tf.concat([next_target, zeros_tokens], axis=1)
            tokens.append(next_token)
            
            # Add zeros for sequences that are done
            if tf.reduce_all(done):
                remaining_iterations = max_length - iteration - 1
                zeros_remaining = tf.zeros([batch_size, remaining_iterations], dtype=tf.int64)
                tokens.append(zeros_remaining)
                break
        
        tokens = tf.concat(tokens, axis=1)  # Concatenate along the sequence length dimension
        return tokens           



class AutoregressiveTransformer(tf.keras.Model):
    """
    An implementation of an Abstractor-based Transformer module.

    This supports several architectures, including:
    a) X -> Abstractor -> Decoder -> Y
    b) X -> Encoder -> Abstractor -> Decoder -> Y
    c) X -> [Encoder, Abstractor] -> Decoder -> Y
    d) X -> Encoder -> Abstractor; [Encoder, Abstractor] -> Decoder -> Y
    """
    def __init__(self,
            encoder_kwargs,
            abstractor_kwargs,
            decoder_kwargs,
            input_vocab,
            target_vocab,
            embedding_dim,
            output_dim,
            word_to_id,
            abstractor_type='relational', # 'abstractor', 'simple', 'relational', or 'symbolic'
            abstractor_on='input', # 'input' or 'encoder'
            decoder_on='abstractor', # 'abstractor' or 'encoder-abstractor'
            name=None):
        """Creates an autoregressive Abstractor model.

        Parameters
        ----------
        encoder_kwargs : dict
            kwargs for the Encoder module. Can be set to None if architecture does not use an encoder.
        abstractor_kwargs : dict
            kwargs for the Abstractor model. Should match `abstractor_type`
        decoder_kwargs : dict
            kwargs for Decoder module.
        input_vocab : int or 'vector'
            if input is tokens, the size of vocabulary as an int.
            if input is vectors, the string 'vector'. used to create embedder.
        target_vocab : int or 'vector'
            if input is tokens, the size of vocabulary as an int.
            if input is vectors, the string 'vector'. used to create embedder.
        embedding_dim : int or tuple[int]
            dimension of embedding (input will be transformed to this dimension).
        output_dim : int
            dimension of final output. e.g.: # of classes.
        abstractor_type : 'abstractor', 'relational', or 'symbolic', optional
            The type of Abstractor to use, by default 'relational'
        abstractor_on: 'input' or 'encoder'
            what the abstractor should take as input.
        decoder_on: 'abstractor' or 'encoder-abstractor'
            what should form the decoder's 'context'.
            if 'abstractor' the context is the output of the abstractor.
            if 'encoder-abstractor' the context is the concatenation of the outputs of the encoder and decoder. 
        """

        super().__init__(name=name)

        # set params
        self.relation_on = abstractor_on
        self.decoder_on = decoder_on
        self.abstractor_type = abstractor_type
        self.word_to_id = word_to_id           
        self.end_token = self.word_to_id(';')
        self.start_token = self.word_to_id('@')  
        # if relation is computed on inputs and the decoder attends only to the abstractor,
        # there is no need for an encoder
        if (abstractor_on, decoder_on) == ('input', 'abstractor'):
            self.use_encoder = False
            print(f'NOTE: no encoder will be used since relation_on={abstractor_on} and decoder_on={decoder_on}')
        else:
            self.use_encoder = True

        # set up source and target embedders
        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.Dense(embedding_dim, name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.Dense(embedding_dim, name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        # initialize layers
        if self.use_encoder:
            self.encoder = Encoder(**encoder_kwargs, name='encoder')

        # initialize the abstractor based on requested type
        if abstractor_type == 'abstractor':
            self.abstractor = Abstractor(**abstractor_kwargs, name='abstractor')
        elif abstractor_type == 'relational':
            self.abstractor = RelationalAbstracter(**abstractor_kwargs, name='abstractor')
        elif abstractor_type == 'symbolic':
            self.abstractor = SymbolicAbstracter(**abstractor_kwargs, name='abstractor')
        else:
            raise ValueError(f'unexpected `abstracter_type` argument {abstractor_type}')

        # initialize decoder
        self.decoder = MultiAttentionDecoder(**decoder_kwargs, name='decoder')

        # initialize final prediction layer
        self.final_layer = layers.Dense(output_dim, name='final_layer')


    def call(self, inputs):
        source, target = inputs # get source and target from inputs

        # embed source and add positional embedding
        source = self.source_embedder(source)
        source = self.pos_embedding_adder_input(source)
        encoder_context = self.encoder(source)


        # embed target and add positional embedding
        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)
        decoder_inputs = [target_embedding, encoder_context]
        x = self.decoder(decoder_inputs)
        # produce final prediction
        logits = self.final_layer(x)
        return logits
        
    def get_initial_state(self, context, max_length):
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        zeros_tensor = tf.zeros([batch_size, max_length - 1], dtype=tf.int64)
        start_target = tf.concat([start_tokens, zeros_tensor], axis=1)
        return start_target, done

    def get_next_token(self, context, target, iteration, done, temperature=0.0):
        logits = self((context, target))
        pred = tf.argmax(logits, axis=-1)
        done = done | (tf.expand_dims(pred[:, iteration], axis=-1) == self.end_token)
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), tf.expand_dims(pred[:, iteration], axis=-1))
        return next_token, done

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.context_text_processor(texts)
        return context

    def translate(self, context, max_length=30, temperature=0.0):
        batch_size = tf.shape(context)[0]
        tokens = []
        next_target, done = self.get_initial_state(context, max_length)
        
        for iteration in range(max_length):
            next_token, done = self.get_next_token(context, next_target, iteration, done)
            zeros_tokens = tf.zeros([batch_size, max_length - iteration - 1], dtype=tf.int64)
            next_target = tf.concat([next_target[:, :iteration + 1], next_token], axis=1)
            next_target = tf.concat([next_target, zeros_tokens], axis=1)
            tokens.append(next_token)
            
            # Add zeros for sequences that are done
            if tf.reduce_all(done):
                remaining_iterations = max_length - iteration - 1
                zeros_remaining = tf.zeros([batch_size, remaining_iterations], dtype=tf.int64)
                tokens.append(zeros_remaining)
                break
        
        tokens = tf.concat(tokens, axis=1)  # Concatenate along the sequence length dimension
        return tokens           




class Encoder_LSTM(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder_LSTM, self).__init__()
        self.enc_units = enc_units
        self.bi_lstm = Bidirectional(LSTM(self.enc_units, return_sequences=True, 
                                          return_state=True, recurrent_initializer='glorot_uniform'))
    
    def call(self, x):
        output, forward_h, forward_c, backward_h, backward_c = self.bi_lstm(x)
        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
        return output, state_h, state_c



class BahdanauAttention(Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
    
    def call(self, query, values):
        # query: decoder hidden state (previous time step)
        # values: encoder outputs
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector
        
class Decoder_LSTM(Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder_LSTM, self).__init__()
        self.dec_units = dec_units
        self.lstm = LSTM(self.dec_units, return_sequences=True, return_state=True, 
                         recurrent_initializer='glorot_uniform')
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.fc = Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)
    
    def call(self, x, hidden, enc_output):
        x = self.embedding(x)
        context_vector = self.attention(hidden, enc_output)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state, _ = self.lstm(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x
        
        
                
class AutoregressiveAttentionalLSTM(tf.keras.Model):
    """
    An implementation of an Abstractor-based Transformer module.

    This supports several architectures, including:
    a) X -> Abstractor -> Decoder -> Y
    b) X -> Encoder -> Abstractor -> Decoder -> Y
    c) X -> [Encoder, Abstractor] -> Decoder -> Y
    d) X -> Encoder -> Abstractor; [Encoder, Abstractor] -> Decoder -> Y
    """
    def __init__(self,
            encoder_kwargs,
            abstractor_kwargs,
            decoder_kwargs,
            input_vocab,
            target_vocab,
            embedding_dim,
            output_dim,
            word_to_id,
            abstractor_type='relational', # 'abstractor', 'simple', 'relational', or 'symbolic'
            abstractor_on='input', # 'input' or 'encoder'
            decoder_on='abstractor', # 'abstractor' or 'encoder-abstractor'
            name=None):
        """Creates an autoregressive Abstractor model.

        Parameters
        ----------
        encoder_kwargs : dict
            kwargs for the Encoder module. Can be set to None if architecture does not use an encoder.
        abstractor_kwargs : dict
            kwargs for the Abstractor model. Should match `abstractor_type`
        decoder_kwargs : dict
            kwargs for Decoder module.
        input_vocab : int or 'vector'
            if input is tokens, the size of vocabulary as an int.
            if input is vectors, the string 'vector'. used to create embedder.
        target_vocab : int or 'vector'
            if input is tokens, the size of vocabulary as an int.
            if input is vectors, the string 'vector'. used to create embedder.
        embedding_dim : int or tuple[int]
            dimension of embedding (input will be transformed to this dimension).
        output_dim : int
            dimension of final output. e.g.: # of classes.
        abstractor_type : 'abstractor', 'relational', or 'symbolic', optional
            The type of Abstractor to use, by default 'relational'
        abstractor_on: 'input' or 'encoder'
            what the abstractor should take as input.
        decoder_on: 'abstractor' or 'encoder-abstractor'
            what should form the decoder's 'context'.
            if 'abstractor' the context is the output of the abstractor.
            if 'encoder-abstractor' the context is the concatenation of the outputs of the encoder and decoder. 
        """

        super().__init__(name=name)

        # set params
        self.relation_on = abstractor_on
        self.decoder_on = decoder_on
        self.abstractor_type = abstractor_type
        self.word_to_id = word_to_id           
        self.end_token = self.word_to_id(';')
        self.start_token = self.word_to_id('@')  
        self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')

        self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')
        self.decoder = Decoder_LSTM(target_vocab, embedding_dim, dec_units=128)
        self.encoder = Encoder_LSTM(input_vocab, embedding_dim, enc_units=32)
        self.final_layer = layers.Dense(output_dim, name='final_layer')


    def call(self, inputs):
        source, target = inputs # get source and target from inputs

        # embed source and add positional embedding
        source = self.source_embedder(source)
        source = self.pos_embedding_adder_input(source)

        encoder_context, hidden_context, state = self.encoder(source)
        
        predictions = []
        for t in range(target.shape[1]):
            dec_input = tf.expand_dims(target[:, t], 1)
            prediction = self.decoder(dec_input, hidden_context, encoder_context)
            predictions.append(prediction)
        logits = tf.stack(predictions, axis=1)
        return logits
        
    def get_initial_state(self, context, max_length):
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        zeros_tensor = tf.zeros([batch_size, max_length - 1], dtype=tf.int64)
        start_target = tf.concat([start_tokens, zeros_tensor], axis=1)
        return start_target, done

    def get_next_token(self, context, target, iteration, done, temperature=0.0):
        logits = self((context, target))
        pred = tf.argmax(logits, axis=-1)
        done = done | (tf.expand_dims(pred[:, iteration], axis=-1) == self.end_token)
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), tf.expand_dims(pred[:, iteration], axis=-1))
        return next_token, done

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.context_text_processor(texts)
        return context

    def translate(self, context, max_length=30, temperature=0.0):
        batch_size = tf.shape(context)[0]
        tokens = []
        next_target, done = self.get_initial_state(context, max_length)
        
        for iteration in range(max_length):
            next_token, done = self.get_next_token(context, next_target, iteration, done)
            zeros_tokens = tf.zeros([batch_size, max_length - iteration - 1], dtype=tf.int64)
            next_target = tf.concat([next_target[:, :iteration + 1], next_token], axis=1)
            next_target = tf.concat([next_target, zeros_tokens], axis=1)
            tokens.append(next_token)
            
            # Add zeros for sequences that are done
            if tf.reduce_all(done):
                remaining_iterations = max_length - iteration - 1
                zeros_remaining = tf.zeros([batch_size, remaining_iterations], dtype=tf.int64)
                tokens.append(zeros_remaining)
                break
        
        tokens = tf.concat(tokens, axis=1)  # Concatenate along the sequence length dimension
        return tokens                  
        


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
        output = tf.matmul(inputs, weights) 
        return output
    


class AnotherSmallConvNet(tf.keras.Model):
    def __init__(self):
        super(AnotherSmallConvNet, self).__init__()
        self.conv1 = BipolarDense(1024)
        self.dropout = layers.Dropout(0.3)
        self.ln = layers.LayerNormalization()
    def call(self, x):
        #x = tf.expand_dims(x, axis=2)
        x = self.conv1(x)  
        #x = tf.squeeze(x, axis=2)
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
        #self.encoder = ConvLayerWithLearnableH(S = 160, h_size=1024-128+1)
        self.conv1 = layers.Conv1D(filters=128, kernel_size=5, strides=3, padding='same')
        self.conv2 = layers.Conv1D(filters=1024, kernel_size=3, strides=5, padding='same')

        self.dropout = layers.Dropout(0.4)
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def call(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = tf.nn.silu(x)

        x = self.conv2(x)
        x = self.ln2(x)
        x = tf.nn.silu(x)
        
        #x = self.encoder(x)
        #x = self.ln1(x)
        #x = tf.nn.silu(x)     
        x = self.dropout(x, training=True)
        return x


class HDSymbolicAttention(layers.Layer):
    def __init__(self, d_model, dim, seq_len_s, embd_size, symbolic, name="hd_symbolic_attention", **kwargs):
        super(HDSymbolicAttention, self).__init__(name=name, **kwargs)
        self.d_model = d_model
        self.dim = dim
        self.n_seq = seq_len_s
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
        self.Symbols = tf.Variable(normal_initializer(shape=(input_shape[1], self.embd_size)), name='symbols', trainable=False)
        super(HDSymbolicAttention, self).build(input_shape)

    def call(self, values):
        self.S3 = tf.zeros_like(values)
        values_projected = self.process(values)
        symbol_projected = self.process(self.Symbols+self.S3)   
        
        if self.symbolic:
            scores = self.create_cosine_similarity_matrix(values_projected,symbol_projected)
        else:
            scores = self.create_cosine_similarity_matrix(values_projected,values_projected) 
        attention_output = tf.matmul(scores,values_projected)
        O = tf.nn.silu(attention_output*symbol_projected)
        V = attention_output
        return O, values_projected

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.d_model)



class CustomLinear(tf.keras.layers.Layer):
    def __init__(self, T, D):
        super(CustomLinear, self).__init__()
        # Define the IDs buffer (not trainable)
        self.IDs = tf.random.normal((D, T)) * 0.01
        # Define the weight matrix (trainable parameter)
        self.W = self.add_weight(shape=(D, T), initializer=tf.random_normal_initializer(stddev=0.1), trainable=True)
        self.Y_n = self.W

    def call(self, x):
        # Element-wise multiplication with IDs and normalize
        Y = self.W * self.IDs
        Y_n = Y / tf.norm(Y)
        self.Y_n = Y_n
        # Perform matrix multiplication and return result
        Res = tf.matmul(x, tf.transpose(Y_n))
        return Res


class MultivariateSeq2SeqModel(tf.keras.Model):
    def __init__(self, T, D):
        super(MultivariateSeq2SeqModel, self).__init__()
        self.fc1 = CustomLinear(T, D)
        self.eib = self.add_weight(shape=(160,D), initializer=tf.random_normal_initializer(stddev=0.01), trainable=True)
        self.out = self.add_weight(shape=(160,10), initializer=tf.random_normal_initializer(stddev=0.01), trainable=True)
        self.oib = self.add_weight(shape=(10,D), initializer=tf.random_normal_initializer(stddev=0.01), trainable=True)
        self.actv = tf.keras.activations.mish  # Mish activation

    def encode(self, x):
        h = self.fc1(x)
        h = tf.einsum('bij,ij->bij',h,self.eib)
        h = tf.einsum('bij,ik->bkj',h,self.out)
        h = tf.einsum('bij,ij->bij',h,self.oib)
        return self.actv(h)

    def call(self, x_seq):
        # Forward pass
        h = self.encode(x_seq)
        return h
        
        
        
        
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
        
        self.source_embedder = layers.Embedding(input_vocab, embedding_dim, mask_zero=True, name='source_embedder')
        self.target_embedder = layers.Embedding(target_vocab, embedding_dim, mask_zero=True, name='target_embedder')        
        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.dropout = layers.Dropout(0.3, name='dropout0') 
        self.encoder = Encoder(**encoder_kwargs, name='encoder')
        self.hd_encode = MultivariateSeq2SeqModel(embedding_dim, VSA_dim)
        self.ln = layers.GroupNormalization(epsilon=1e-7)
    def call(self, inputs):
        source, target = inputs
        source = self.source_embedder(source)
        source = self.pos_embedding_adder_input(source)
        source = self.encoder(source)
        source = self.dropout(source)
        source = self.hd_encode(source)
        source = self.ln(source)
        return source



                  
  
class AutoregressiveHDFormer(tf.keras.Model):                
    def __init__(self,
            encoder_kwargs,
            decoder_kwargs,
            input_vocab,
            target_vocab,
            embedding_dim,
            VSA_dim,
            seq_len_s,
            seq_len_t,
            dropout_rate,
            scale_source,
            scale_target,
            symbolic,
            word_to_id,
            name=None):
                
        super().__init__(name=name)
        dropout_rate = 0.4
        self.word_to_id = word_to_id
        mask_token = self.word_to_id('[UNK]')
        self.end_token = self.word_to_id(';')
        self.start_token = self.word_to_id('@') 
        self.bnd = Binding(mask_token, encoder_kwargs, input_vocab, target_vocab, embedding_dim, VSA_dim, scale_source, scale_target, name='binding')
        self.final_layer = layers.Dense(target_vocab)

    def call(self, inputs):
        h_o = self.bnd(inputs)       
        logits = self.final_layer(h_o)
        return logits                                  

          

