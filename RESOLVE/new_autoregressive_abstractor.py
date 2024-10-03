import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import layers
from transformer_modules import Encoder, AddPositionalEmbedding
from multi_attention_decoder import MultiAttentionDecoder
from abstractor import Abstractor
from abstracters import RelationalAbstracter, SymbolicAbstracter
from seq2seq_abstracter_models import HDSymoblicAttention
import tensorflow as tf
import numpy as np


class BipolarDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None):
        super(BipolarDense, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
        )

    def call(self, inputs):
        # Make weights bipolar
        output = tf.matmul(inputs, self.kernel) + self.bias
        if self.activation is not None:
            return self.activation(output)
        return output
        
        
        
def get_angles(pos, i, d_model):
    angle_rates = 1 / tf.math.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return tf.cast(pos, tf.float32) * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                            d_model)
    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])
    
    angle_rads = tf.concat([tf.reshape(sines, (-1, sines.shape[1], 1)), 
                            tf.reshape(cosines, (-1, cosines.shape[1], 1))], axis=-1)
    angle_rads = tf.reshape(angle_rads, (position, d_model))

    pos_encoding = angle_rads[tf.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += positional_encoding(maxlen, self.d_model)
        return x
        

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, seq_l):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = 1
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.wq = BipolarDense(d_model)
        self.wk = BipolarDense(d_model)
        self.wv = BipolarDense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        self.anchor = None
    def build(self, input_shape):
        input_shape0, input_shape1, input_shape2, input_shape3 = input_shape
        self.anchor = self.add_weight(
            shape=(input_shape2[1], self.d_model),
            initializer='glorot_uniform',
            trainable=True,
        )
        
    def scaled_dot_product_attention(self, q, k, v, mask):
        if q.shape[1]==k.shape[2]:
            q_ext = tf.math.sign(q[:, tf.newaxis, :, :])  
            k_ext = tf.math.sign(k[:, :, tf.newaxis, :] +  k[:, tf.newaxis, :, :])
        else:
            q_ext = tf.math.sign(q[:, tf.newaxis, :, :]) 
            k_ext = tf.math.sign(k[:, :, tf.newaxis, :]) +  q_ext 
        scaled_attention_logits = tf.reduce_sum(k_ext * q_ext , axis=-1) / 1024    
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        if mask is not None:
            attention_weights *= mask        
        output = tf.matmul(tf.transpose(attention_weights,perm=[0,2,1]), v)
        output = tf.nn.tanh(output * self.anchor)
        return output, attention_weights
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        v, k, q, mask = inputs
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        #scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights
 
 
class PointWiseFeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, dff, d_model):
        super(PointWiseFeedForwardNetwork, self).__init__()
        self.dense1 = BipolarDense(d_model, activation='relu')
        self.dense2 = BipolarDense(dff)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x    

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, seq_t, seq_e, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads, seq_t)
        self.mha2 = MultiHeadAttention(d_model, num_heads, seq_t)
        self.ffn = PointWiseFeedForwardNetwork(dff, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1((x, x, x, look_ahead_mask))
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2((enc_output, enc_output, out1, padding_mask))
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output)
        return out3, attn_weights_block1, attn_weights_block2




class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, seq_e, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, seq_e)
        self.ffn = PointWiseFeedForwardNetwork(dff, d_model)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha((x, x, x, mask))
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(ffn_output)
        return out2
        

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, seq_e, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = TokenEmbedding(input_vocab_size, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, seq_e, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, training=True):
        x = self.embedding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x  
                                

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, seq_e, seq_t, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, seq_e, seq_t, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
        return x, attention_weights
        



class AutoregressiveHDFormer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, seq_t, seq_e, word_to_id, rate=0.1):
        super(AutoregressiveHDFormer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, seq_e, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, seq_e, seq_t, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self.word_to_id = word_to_id           
        self.end_token = self.word_to_id(';')
        self.start_token = self.word_to_id('@') 

    def create_padding_mask(self, seq):
        # seq shape: (batch_size, seq_len)
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, :]  # (batch_size, 1, seq_len)
    
    def create_look_ahead_mask(self, size):
        # Mask to hide future tokens
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (size, size)

    def create_look_ahead_mask_mixed(self, size_t, size_i):
        # Mask to hide future tokens
        mask = 1 - tf.linalg.band_part(tf.ones((size_i, size_t)), -1, 0)
        return mask  # (size, size)        
        
    def create_masks(self, inp, tar):
        enc_padding_mask = self.create_padding_mask(inp)
        dec_padding_mask = self.create_look_ahead_mask_mixed(tf.shape(tar)[1],tf.shape(inp)[1])
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        return enc_padding_mask, look_ahead_mask, dec_padding_mask
    
    def call(self, inputs):
        inp, tar = inputs
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        enc_output = self.encoder(inp, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output


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
        done = done | (tf.expand_dims(pred[:, iteration],axis=-1) == self.end_token)
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), tf.expand_dims(pred[:, iteration],axis=-1))
        
        return next_token, done

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.context_text_processor(texts)
        return context
  
    def translate(self,
                context, *,
                max_length=30,
                temperature=0.0):
        batch_size = tf.shape(context)[0]
        tokens = []
        next_target, done = self.get_initial_state(context,max_length)        
        for iteration in range(max_length - 1):
            if max_length - iteration - 3 > 0:
                next_token, done = self.get_next_token(context, next_target, iteration, done)
                zeros_tokens = tf.zeros([batch_size, max_length - iteration - 2], dtype=tf.int64)
                next_target = tf.concat([next_target[:, :iteration + 1], next_token], axis=1)
                next_target = tf.concat([next_target, zeros_tokens], axis=1)
            tokens.append(next_token)
        tokens = tf.concat(tokens, axis=-1)
        return tokens
