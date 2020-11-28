from __future__ import absolute_import

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU
import tensorflow as tf
import numpy as np

class NR_GraphAttention(Layer):

    def __init__(self,
                 node_size,
                 rel_size,
                 triple_size,
                 depth = 1,
                 use_w = False,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.3,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')
        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias
        self.use_w = use_w
        self.depth = depth

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        self.biases = []        
        self.attn_kernels = []  
        self.gat_kernels = []
        self.interfaces = []
        self.gate_kernels = []

        super(NR_GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        node_F = input_shape[0][-1]
        rel_F = input_shape[1][-1]
        self.ent_F = node_F
        ent_F = self.ent_F
            
        
        for l in range(self.depth):
            
            self.attn_kernels.append([])
            for head in range(self.attn_heads):                
                attn_kernel = self.add_weight(shape=(3*node_F ,1),
                                       initializer=self.attn_kernel_initializer,
                                       regularizer=self.attn_kernel_regularizer,
                                       constraint=self.attn_kernel_constraint,
                                       name='attn_kernel_self_{}'.format(head))
                    
                self.attn_kernels[l].append(attn_kernel)
                
        self.built = True
        
    
    def call(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]     
        adj = tf.SparseTensor(K.cast(K.squeeze(inputs[2],axis = 0),dtype = "int64"),
                         K.ones_like(inputs[2][0,:,0]),(self.node_size,self.node_size))
        sparse_indices = tf.squeeze(inputs[3],axis = 0)  
        sparse_val = tf.squeeze(inputs[4],axis = 0)
        
        features = self.activation(features)
        outputs.append(features)
                        
        for l in range(self.depth):
            features_list = []
            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels[l][head]  
                rels_sum = tf.SparseTensor(indices=sparse_indices,values=sparse_val,dense_shape=(self.triple_size,self.rel_size))
                
                rels_sum = tf.sparse_tensor_dense_matmul(rels_sum,rel_emb)
                neighs = K.gather(features,adj.indices[:,1])
                selfs = K.gather(features,adj.indices[:,0])
                
                rels_sum = tf.nn.l2_normalize(rels_sum, 1)
                bias = tf.reduce_sum(neighs * rels_sum, 1, keepdims=True) * rels_sum
                neighs = neighs - 2 * bias
                
                att = K.squeeze(K.dot(K.concatenate([selfs,neighs,rels_sum]),attention_kernel),axis = -1)
                att = tf.SparseTensor(indices=adj.indices, values=att, dense_shape=adj.dense_shape)
                att = tf.sparse_softmax(att)
            
                new_features = tf.segment_sum (neighs*K.expand_dims(att.values,axis = -1),adj.indices[:,0])
                features_list.append(new_features)

            if self.attn_heads_reduction == 'concat':
                features = K.concatenate(features_list)  # (N x KF')
            else:
                features = K.mean(K.stack(features_list), axis=0)

            features = self.activation(features)
            outputs.append(features)
        
        outputs = K.concatenate(outputs)
        return outputs

    def compute_output_shape(self, input_shape):    
        node_shape = self.node_size, (input_shape[0][-1]) * (self.depth+1)
        return node_shape