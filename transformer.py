import numpy as np
import tensorflow as tf
import transformer_func as transformer

class Transformer_Seq2Seq(tf.keras.Model):

    def __init__(self):
        super(Transformer_Seq2Seq, self).__init__()
        self.window_size = 20
        
        self.lr = 0.0005
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.batch_size = 100
        self.embedding_size = 7056	
        self.pos_encoder = transformer.Position_Encoding_Layer(self.window_size, self.embedding_size)
        self.encoder1 = transformer.Transformer_Block(self.embedding_size, False, multi_headed=False)
        self.fc1 = tf.keras.layers.Dense(self.embedding_size)
        
    @tf.function
    def call(self, encoder_input):
        pos_embed = self.pos_encoder(encoder_input)
        encoder_output = self.encoder1(pos_embed)
        return self.fc1(encoder_output)

    # def accuracy_function(self, prbs, labels, mask):
    #     decoded_symbols = tf.argmax(input=prbs, axis=2)
    #     accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
    #     return accuracy

    # def loss_function(self, prbs, labels, mask):
    #     loss_fxn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    #     loss = loss_fxn(labels, prbs, sample_weight=mask)
    #     return tf.math.reduce_sum(loss)