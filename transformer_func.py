import tensorflow as tf
import numpy as np
import attenvis as av

def Attention_Matrix(K, Q, use_mask=False):
    window_size_queries = Q.get_shape()[1] # window size of queries

    window_size_keys = K.get_shape()[1] # window size of keys

    mask = tf.convert_to_tensor(value=np.transpose(np.tril(np.ones((window_size_queries,window_size_keys))*np.NINF,-1),(1,0)),dtype=tf.float32)

    atten_mask = tf.tile(tf.reshape(mask,[-1,window_size_queries,window_size_keys]),[tf.shape(input=K)[0],1,1])

    # 1) compute attention weights using queries and key matrices (if use_mask==True, then make sure to add the attention mask before softmax)
    dotprod = tf.math.reduce_sum(tf.multiply(K, Q), axis=2, keepdims=True)
    embed_dim = tf.cast(K.shape[2], dtype=tf.float32)
    prod  = dotprod / tf.math.sqrt(embed_dim)
    if use_mask:
        prod = tf.boolean_mask(prod, mask)
    prod = tf.nn.softmax(prod)
    return prod

class Atten_Head(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, use_mask):		
        super(Atten_Head, self).__init__()
        self.use_mask = use_mask
        self.w_K = self.add_weight(shape=(input_size, output_size), initializer="random_normal", trainable=True)
        self.w_V = self.add_weight(shape=(input_size, output_size), initializer="random_normal", trainable=True)
        self.w_Q = self.add_weight(shape=(input_size, output_size), initializer="random_normal", trainable=True)

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        # - Apply 3 matrices to turn inputs into keys, values, and queries. You will need to use tf.tensordot for this.
        k_out = tf.tensordot(inputs_for_keys, self.w_K, [[2],[1]])
        v_out = tf.tensordot(inputs_for_values, self.w_V, [[2], [1]])
        q_out = tf.tensordot(inputs_for_queries, self.w_Q, [[2], [1]])
        # - Call Attention_Matrix with the keys and queries, and with self.use_mask.
        attention = Attention_Matrix(k_out, q_out)
        # - Apply the attention matrix to the values
        return tf.multiply(v_out, attention)

class Multi_Headed(tf.keras.layers.Layer):

    def __init__(self, emb_sz, use_mask):
        super(Multi_Headed, self).__init__()
        self.head_size = emb_sz // 3
        remainder = emb_sz - (self.head_size * 2)
        self.head1 = Atten_Head(emb_sz // 3, emb_sz // 3, use_mask)
        self.head2 = Atten_Head(emb_sz // 3, emb_sz // 3, use_mask)
        self.head3 = Atten_Head(remainder, remainder, use_mask)
        self.lin_layer = tf.keras.layers.Dense(emb_sz)

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        input1_keys = inputs_for_keys[:,:,0:self.head_size]
        input1_values = inputs_for_values[:,:,0:self.head_size]
        input1_queries = inputs_for_queries[:,:,0:self.head_size]
        output_head1 = self.head1(input1_keys, input1_values, input1_queries)
        input2_keys = inputs_for_keys[:,:,self.head_size:self.head_size*2]
        input2_queries = inputs_for_queries[:,:,self.head_size:self.head_size*2]
        input2_values = inputs_for_values[:,:,self.head_size:self.head_size*2]
        output_head2 = self.head2(input2_keys, input2_values, input2_queries)
        input3_keys = inputs_for_keys[:,:,self.head_size*2:]
        input3_queries = inputs_for_queries[:,:,self.head_size*2:]
        input3_values = inputs_for_values[:,:,self.head_size*2:]
        output_head3 = self.head3(input3_keys, input3_values, input3_queries)
        concat_output = tf.concat([output_head1, output_head2, output_head3], axis=2)
        return self.lin_layer(concat_output)

class Feed_Forwards(tf.keras.layers.Layer):
	def __init__(self, emb_sz):
		super(Feed_Forwards, self).__init__()
		self.layer_1 = tf.keras.layers.Dense(emb_sz,activation='relu')
		self.layer_2 = tf.keras.layers.Dense(emb_sz)


	@tf.function
	def call(self, inputs):
		layer_1_out = self.layer_1(inputs)
		layer_2_out = self.layer_2(layer_1_out)
		return layer_2_out


class Transformer_Block(tf.keras.layers.Layer):
    def __init__(self, emb_sz, is_decoder, multi_headed=False):
        super(Transformer_Block, self).__init__()
        self.ff_layer = Feed_Forwards(emb_sz)
        self.self_atten = Atten_Head(emb_sz,emb_sz,use_mask=is_decoder) if not multi_headed else Multi_Headed(emb_sz,use_mask=is_decoder)
        self.is_decoder = is_decoder

        if self.is_decoder:
            self.self_context_atten = Atten_Head(emb_sz,emb_sz,use_mask=False) if not multi_headed else Multi_Headed(emb_sz,use_mask=False)
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)


    @tf.function
    def call(self, inputs, context=None):
        with av.trans_block(self.is_decoder):
            atten_out = self.self_atten(inputs,inputs,inputs)
        atten_out+=inputs
        atten_normalized = self.layer_norm(atten_out)

        if self.is_decoder:
            assert context is not None,"Decoder blocks require context"
            context_atten_out = self.self_context_atten(context,context,atten_normalized)
            context_atten_out+=atten_normalized
            atten_normalized = self.layer_norm(context_atten_out)

        ff_out=self.ff_layer(atten_normalized)
        ff_out+=atten_normalized
        ff_norm = self.layer_norm(ff_out)
        return tf.nn.relu(ff_norm)


class Position_Encoding_Layer(tf.keras.layers.Layer):
	def __init__(self, window_sz, emb_sz):
		super(Position_Encoding_Layer, self).__init__()
		self.positional_embeddings = self.add_weight("pos_embed",shape=[window_sz, emb_sz])

	@tf.function
	def call(self, x):
		return x+self.positional_embeddings
