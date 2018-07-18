# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 09:57:64 2018

@author: junjiao
"""

# =========================================================================================
# There are some notations.
# N is batch size.
# K is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
#
# =========================================================================================

from __future__ import division
import tensorflow as tf
slim = tf.contrib.slim
from core.utils import *

DEVICE = 'big' #'big'
if DEVICE == 'hinton':
  data_folder = '/data'
else:
  data_folder = '/data1'

class CaptionGenerator(object):
    def __init__(self, word_to_idx, dim_feature=[64, 2048], dim_embed=512, dim_hidden=512, n_time_step=16,
                 alpha_c=0.0, dropout=True):
        """
        if spatially adaptive pooling/ spatial pyramid pooling
             (14,14,2048)
        else
            (7,7,2048)
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of Resnet last conv feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.alpha_c = alpha_c
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.K = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        #self.features = tf.placeholder(tf.float32, [None, self.K, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        self.decision_label = tf.placeholder(tf.float32,[None,self.T+1])
        self.attributes = tf.placeholder(tf.int32,[None,30])
        self.attributes_onehot = tf.placeholder(tf.float32,[None,1113])
        self.keep_prob = tf.placeholder(tf.float32)
        self.attribute_list = load_pickle(data_folder+'/junjiaot/data/train/multi_class_labels_list.pkl')

    def _get_initial_lstm(self, features):
        '''
        features: (N,K,H)
        '''
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1) #(N,H)

            w_h = tf.get_variable('w_h', [self.H, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.H, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h


    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x


    def _image_features(self, features ,global_feature = True,reuse = False):
        '''
        # features (N,K,H)
        # v_i = ReLu(W_a * a_i),V_g = ReLu(W_b * a_g)
        #I do not want to noisy global image representation instead
        # I would like to use high level concepts to initialize the network
        '''
        with tf.variable_scope('image_features',reuse=reuse):
            #features_t = tf.nn.dropout(features, self.keep_prob)
            w_a = tf.get_variable('w_a', [self.D, self.H], initializer=self.weight_initializer)
            b_a = tf.get_variable('b_a', [self.H], initializer=self.const_initializer)
            features_flat = tf.reshape(features, [-1, self.D])      #(N*K, D)
            features_proj = tf.nn.relu(tf.matmul(features_flat, w_a)+b_a) #(N*K, H)
            features_proj = tf.reshape(features_proj, [-1, self.K, self.H])     #(N,K,H)
            if global_feature == True:
                w_b = tf.get_variable('w_b', [self.D, self.M], initializer=self.weight_initializer)
                b_b = tf.get_variable('b_b', [self.M], initializer=self.const_initializer)
                features_global = tf.reduce_mean(features,axis=1) #(N,D) average pooling
                features_global = tf.nn.dropout(features_global, self.keep_prob)
                features_global = tf.nn.relu(tf.matmul(features_global, w_b)+b_b,name = 'global_feature') #(N,M)
                return features_proj,features_global
        return features_proj


    def _attention_layer(self, features_proj, features_orig ,h, vs,reuse=False):
        '''
        # W_v * V + W_g * h_t
        # h (N,H); features_proj (N,K,H); vs (N,H)
        # c (N,H)
        '''
        with tf.variable_scope('attention_layer', reuse=reuse):
            w_v = tf.get_variable('w_v', [self.H, 512], initializer=self.weight_initializer)
            w_g = tf.get_variable('w_g', [self.H, 512], initializer=self.weight_initializer)
            w_s = tf.get_variable('w_s', [self.H, 512], initializer=self.weight_initializer)
            w_h = tf.get_variable('w_h', [512, 1], initializer=self.weight_initializer)
            #w_g2 = tf.get_variable('w_g2', [self.H, self.K], initializer=self.weight_initializer)
            #w_v2 = tf.get_variable('w_v2', [self.H, self.K], initializer=self.weight_initializer)

            h_t = tf.nn.dropout(h, self.keep_prob)
            features_proj = tf.nn.dropout(features_proj,self.keep_prob)

            temp_v = tf.matmul(tf.reshape(features_proj,[-1,self.H]),w_v)   #(N*K,512)
            temp_v = tf.reshape(temp_v,[-1,self.K,512]) #(N,K,512)
            content_v = tf.nn.tanh(temp_v + tf.expand_dims(tf.matmul(h_t,w_g),1)) #(N,K,512)

            #temp_v2 = tf.matmul(tf.reshape(features_proj,[-1,self.H]),w_v2)   #(N*K,K)
            #temp_v2 = tf.reshape(temp_v2,[-1,self.K,self.K]) #(N,K,K)
            #content_v = tf.multiply(content_v,tf.nn.sigmoid(temp_v2 + tf.expand_dims(tf.matmul(h_t,w_g2),2)))

            content_v = tf.nn.dropout(content_v, self.keep_prob)
            z_t = tf.reshape(tf.matmul(tf.reshape(content_v,[-1,512]),w_h),[-1,self.K]) #(N,K)
            alpha = tf.nn.softmax(z_t) #(N,K)
            c = tf.reduce_sum(features_proj * tf.expand_dims(alpha, 2), 1, name='context') #(N,H)
            context_vector_full = tf.reduce_sum(features_orig * tf.expand_dims(alpha, 2), 1, name='context_full')

            vs = tf.nn.dropout(vs, self.keep_prob)
            h_t = tf.nn.dropout(h, self.keep_prob)
            content_s = tf.nn.tanh(tf.matmul(vs,w_s) + tf.matmul(h_t,w_g)) #(N,K)
            content_s = tf.nn.dropout(content_s, self.keep_prob)
            z_t_extended = tf.matmul(content_s,w_h) #(N,1)
            extended = tf.concat([z_t,z_t_extended],1)
            alpha_hat = tf.nn.softmax(extended) #(N,K+1)
            beta = tf.reshape(alpha_hat[:,-1],[-1,1],name = 'beta') #(N,1)
            #print(beta)
            # c_hat = beta * vs + (1-beta) * c
            #c_hat = tf.multiply(vs,tf.tile(beta,[1,self.H])) + tf.multiply(c,tf.tile((1-beta),[1,self.H])) # (N,H)
            c_hat = tf.multiply(vs,beta) + tf.multiply(c,(1-beta)) # (N,H)

            return context_vector_full,c_hat, alpha, beta


    def _attention_layer2(self, features_proj, features_orig ,h,alpha_in, vs,reuse=False):
        '''
        # W_v * V + W_g * h_t
        # h (N,H); features_proj (N,K,H); vs (N,H)
        # c (N,H)
        what about inverse feature (??????)
        '''
        with tf.variable_scope('the_second_attention_layer', reuse=reuse):
            w_v = tf.get_variable('w_v', [self.H, 512], initializer=self.weight_initializer)
            w_g = tf.get_variable('w_g', [self.H, 512], initializer=self.weight_initializer)
            w_s = tf.get_variable('w_s', [self.H, 512], initializer=self.weight_initializer)
            w_h = tf.get_variable('w_h', [512, 1], initializer=self.weight_initializer)
            #w_g2 = tf.get_variable('w_g2', [self.H, self.K], initializer=self.weight_initializer)
            #w_v2 = tf.get_variable('w_v2', [self.H, self.K], initializer=self.weight_initializer)
            #w_v = tf.stop_gradient(w_v)
            #w_g = tf.stop_gradient(w_g)
            #w_s = tf.stop_gradient(w_s)
            #w_h = tf.stop_gradient(w_h)
            #features_proj = tf.stop_gradient(features_proj)
            #features_orig = tf.stop_gradient(features_orig) 

            alpha_in = tf.stop_gradient(alpha_in)
            h_t = tf.nn.dropout(h, self.keep_prob)
            features_proj = tf.nn.dropout(features_proj,self.keep_prob)
            features_proj_alpha = features_proj * tf.expand_dims(alpha_in, 2)
            temp_v = tf.matmul(tf.reshape(features_proj_alpha,[-1,self.H]),w_v)   #(N*K,512)
            temp_v = tf.reshape(temp_v,[-1,self.K,512]) #(N,K,512)
            content_v = tf.nn.tanh(temp_v + tf.expand_dims(tf.matmul(h_t,w_g),1)) #(N,K,512)

            #temp_v2 = tf.matmul(tf.reshape(features_proj,[-1,self.H]),w_v2)   #(N*K,K)
            #temp_v2 = tf.reshape(temp_v2,[-1,self.K,self.K]) #(N,K,K)
            #content_v = tf.multiply(content_v,tf.nn.sigmoid(temp_v2 + tf.expand_dims(tf.matmul(c_hat_1,w_g2),2)))

            content_v = tf.nn.dropout(content_v, self.keep_prob)
            z_t = tf.reshape(tf.matmul(tf.reshape(content_v,[-1,512]),w_h),[-1,self.K]) #(N,K)
            alpha = tf.nn.softmax(z_t) #(N,K)



            alpha_mask = tf.to_float(tf.greater_equal(alpha,alpha_in))
            alpha = tf.nn.softmax(tf.multiply(alpha_mask,z_t))



            c = tf.reduce_sum(features_proj * tf.expand_dims(alpha, 2), 1, name='context') #(N,H)
            context_vector_full = tf.reduce_sum(features_orig * tf.expand_dims(alpha, 2), 1, name='context_full')

            vs = tf.nn.dropout(vs, self.keep_prob)
            h_t = tf.nn.dropout(h, self.keep_prob)
            content_s = tf.nn.tanh(tf.matmul(vs,w_s) + tf.matmul(h_t,w_g)) #(N,K)
            content_s = tf.nn.dropout(content_s, self.keep_prob)
            z_t_extended = tf.matmul(content_s,w_h) #(N,1)
            extended = tf.concat([z_t,z_t_extended],1)
            alpha_hat = tf.nn.softmax(extended) #(N,K+1)
            beta = tf.reshape(alpha_hat[:,-1],[-1,1],name = 'beta') #(N,1)
            c_hat = tf.multiply(vs,beta) + tf.multiply(c,(1-beta)) # (N,H)

            return context_vector_full,c_hat, alpha, beta


    def _visual_sentinel(self,h,c,x,reuse = False):
        '''
        # h (N,H); c (N,H); x (N,M+H)
        '''
        input_size = self.M+self.M # if use word embedding plus global feauture
        with tf.variable_scope('Visual_Sentinel', reuse=reuse):
            w_x = tf.get_variable('w_x', [input_size, self.H], initializer=self.weight_initializer)
            w_h = tf.get_variable('w_h', [self.H, self.H], initializer=self.weight_initializer)
            x = tf.nn.dropout(x, self.keep_prob)
            h = tf.nn.dropout(h, self.keep_prob)
            gate_t = tf.matmul(x,w_x) + tf.matmul(h,w_h) #(N,H)
            vs = tf.nn.sigmoid(gate_t)*tf.nn.tanh(c) #(N,H)
        return vs

    def _copying_layer(self,V_c,h,c_hat,detected_prob,dropout = False,reuse = False):
        '''
        V_c (Vc,M)
        h: (N,H); c_hat: (N,H)
        detected_prob: (N,Vc)
        '''
        with tf.variable_scope('copying', reuse=reuse):
            w_c = tf.get_variable('w_c',[self.M,self.H],initializer=self.weight_initializer)
            pr = tf.transpose(tf.matmul(tf.nn.relu(tf.matmul(V_c,w_c)),tf.transpose(h+c_hat)))  #(N,Vc)
            logits = pr * detected_prob #(N,Vc)
        return logits


    def _decode_layer(self, h, c_hat,stop_gradient = True,reuse=False):
        '''
        detected_prob attriubtes probability: (Vc,N)
        h: (N,H); c_hat: (N,H)
        '''
        with tf.variable_scope('decode_layer', reuse=reuse):
            h = tf.nn.dropout(h, self.keep_prob)
            c_hat = tf.nn.dropout(c_hat, self.keep_prob)
            #with tf.variable_scope('generative'):
            w_g = tf.get_variable('w_g',[self.H,self.V],initializer=self.weight_initializer)
            #w_h = tf.get_variable('w_h',[self.H,self.V],initializer=self.weight_initializer)
            b_g = tf.get_variable('b_g', [self.V], initializer=self.const_initializer)
            if stop_gradient:
                w_g = tf.stop_gradient(w_g)
                b_g = tf.stop_gradient(b_g)
            #w = tf.get_variable('w',[2048,self.V],initializer=self.weight_initializer)
            #b = tf.get_variable('b', [self.V], initializer=self.const_initializer)
            #logits = tf.nn.relu(tf.matmul(h+c_hat,w_g)+b_g)
            #logits = tf.nn.dropout(logits,self.keep_prob)
            logits = tf.matmul(c_hat+h,w_g) + b_g
            #logits = tf.matmul(c_hat,w_g) + b_g
        return logits

    def _inverse_word_emb(self,word_vector,reuse=False):
        with tf.variable_scope('word_embedding', reuse=True):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            return tf.matmul(word_vector,tf.transpose(w))



    def _static_attribute_attention_layer(self,features,reuse = False):
      with tf.variable_scope('MIL',reuse = reuse):
        #with tf.device('/gpu:0'):
        w_a = tf.get_variable('w_a', [1024, 1], initializer=tf.contrib.layers.xavier_initializer())
        b_a = tf.get_variable('b_a', [1], initializer=tf.constant_initializer(0.0))
        v = tf.get_variable('v', [2048, 1024], initializer=tf.contrib.layers.xavier_initializer()) #self.H
        u = tf.get_variable('u', [2048, 1024], initializer=tf.contrib.layers.xavier_initializer()) #self.H

        fc_flattened = tf.reshape(features,[-1,2048]) #(c*324,2048)
        fc_flattened = tf.nn.dropout(fc_flattened, self.keep_prob)

        temp_v = tf.nn.tanh(tf.matmul(fc_flattened,v)) # (c*64,1024)
        temp_v = tf.nn.dropout(temp_v, self.keep_prob)

        temp_u = tf.nn.sigmoid(tf.matmul(fc_flattened,u)) # (c*64,1024)
        temp_u = tf.nn.dropout(temp_u, self.keep_prob)

        alpha_logit = tf.reshape(tf.matmul(tf.multiply(temp_v,temp_u),w_a)+b_a,[-1,64,1])

        alpha = tf.nn.softmax(alpha_logit,dim=1) #(c,64,1)

        z = tf.reduce_sum(alpha * tf.reshape(fc_flattened,[-1,64,2048]),1) #(c,512)
        z = tf.nn.dropout(z,self.keep_prob)
        return z,tf.squeeze(alpha,2)

    def _MIL_attention(self,beta,c,z,inital = False,reuse = False):
      with tf.variable_scope('MIL', reuse=reuse):
         w = tf.get_variable('w', [2048, 1113], initializer=tf.contrib.layers.xavier_initializer())
         b = tf.get_variable('b', [1113], initializer=tf.constant_initializer(0.0))
         w_2 = tf.get_variable('w_2', [2048, 2048], initializer=tf.contrib.layers.xavier_initializer())
         b_2 = tf.get_variable('b_2', [2048], initializer=tf.constant_initializer(0.0))
         #print(z.shape,c.shape)
         if not inital:
            w = tf.stop_gradient(w)
            b = tf.stop_gradient(b)
            w_2 = tf.stop_gradient(w_2)
            b_2 = tf.stop_gradient(b_2)
            z = tf.stop_gradient(z)
         z_hat = tf.multiply(z,beta) + tf.multiply(c,(1-beta)) # (N,H)
         z_hat = tf.nn.dropout(z_hat,self.keep_prob)
         temp_logit = tf.nn.relu(tf.matmul(z_hat,w_2)+b_2) #()
         temp_logit = tf.nn.dropout(temp_logit,self.keep_prob)
         logits = tf.matmul(temp_logit,w)+b #(N,1113)
         return logits

    def _MIL_Leaky_noisy_or(self,features,attention_logit,is_training,reuse=False):
        '''
        features: original resenet conv feature
        features :#(N,K,H) -> #(N,7,7,H)
        '''
        with tf.variable_scope('MIL',reuse=reuse):
            fc6 = tf.layers.conv2d(features,filters = 4096,kernel_size = 3,activation = None,padding = 'VALID') #(c,5,5,4096)
            fc6 = slim.batch_norm(fc6, is_training = is_training,activation_fn=tf.nn.relu)

            fc7 = tf.layers.conv2d(fc6,filters = 4096,kernel_size = 1,activation = None)#(c,5,5,4096)
            fc7 = slim.batch_norm(fc7, is_training = is_training,activation_fn=tf.nn.relu)          

            fc8 = tf.layers.conv2d(fc7,filters = 1113,kernel_size = 1,activation = tf.nn.sigmoid,
                               bias_initializer=tf.constant_initializer(-6.8),
                               kernel_initializer=tf.truncated_normal_initializer(stddev = 0.001))#(c,5,5,1113)

            fc8 = tf.reshape(fc8,[-1,36,1113]) #(c,25,1113)

            mil_1 = 1 - fc8 #(c,25,1113)

            max_prob = tf.reduce_max(fc8, axis = 1) #(c,1113)
            prob = 1 - tf.reduce_prod(mil_1,[1])
            final_prob = tf.maximum(max_prob,prob)

            fc8 = tf.stop_gradient(fc8)
            mil_1 = tf.stop_gradient(mil_1)

            leak_prob = tf.nn.sigmoid(attention_logit)
            max_prob2 = tf.maximum(tf.reduce_max(fc8, axis = 1),leak_prob)
            prob2 = 1 - tf.multiply(tf.reduce_prod(mil_1,[1]),1-leak_prob) #(c,1113)
            final_prob2 = tf.maximum(max_prob2,prob2)

        
        return final_prob,final_prob2

    def _attribute_gate_layer(self,beta,c,z,h,x,prob,reuse=False):
        with tf.variable_scope('attribute_attention_layer',reuse=reuse):
            w_z = tf.get_variable('w_z', [self.M,2048 ], initializer=tf.contrib.layers.xavier_initializer())
            b_z = tf.get_variable('b_z', [2048], initializer=tf.constant_initializer(0.0))
            w_z3 = tf.get_variable('w_z3', [2048, 1113], initializer=tf.contrib.layers.xavier_initializer())
            b_z3 = tf.get_variable('b_z3', [1113], initializer=tf.constant_initializer(0.0))
            w_h = tf.get_variable('w_h', [self.H, 2048], initializer=tf.contrib.layers.xavier_initializer())
            prob = tf.stop_gradient(prob)
            z = tf.stop_gradient(z)
            z = tf.nn.dropout(z,self.keep_prob)
            c = tf.nn.dropout(c,self.keep_prob)
            logit_z = tf.nn.relu(tf.matmul(x,w_z) + tf.matmul(h,w_h)+b_z)
            logit_z = tf.nn.dropout(logit_z,self.keep_prob)
            logit_z = tf.matmul(logit_z,w_z3)+b_z3#(N,H)
            attention = tf.multiply(tf.nn.sigmoid(logit_z),prob)
        return attention

    def _attribute_vector_op(self,batch_size,attention_logits,attribute_embedding_expand):
        # =  tf.nn.sigmoid(attention_logits)
        num = tf.reduce_sum(attention_logits,axis = 1,keep_dims = True)
        mask = tf.to_float(tf.equal(num, 0))        
        attribute_attention = attention_logits/(num+mask)#(N,1113)
        attribute_vector = tf.multiply(tf.expand_dims(attribute_attention,2),tf.tile(attribute_embedding_expand,[batch_size,1,1])) #(N,1113,M)
        attribute_vector = tf.reduce_sum(attribute_vector,1) #(N,M)
        return attribute_vector,attribute_attention

    def loss_op(self,logits,labels,pos_weight,model='attention'):
        # Calculate the average cross entropy loss across the batch.
        epsilon = 9e-7
        #pos_weight = 1
        if model == 'noisy_or':
            cross_entropy = -(pos_weight*tf.multiply(labels,tf.log(logits+epsilon))+
                                                          tf.multiply((1-labels),tf.log(1-logits+epsilon)))
        else:
            cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits,pos_weight = pos_weight)
        cross_entropy = tf.reduce_mean(cross_entropy, name='sigmoid_cross_entropy')
        #slim.losses.add_loss(cross_entropy)
        #total_loss = slim.losses.get_total_loss()
        return cross_entropy#cross_entropy_mean
    def _MIL_attention2(self,beta,c,z,reuse = False):
        pass
    def _MIL_Leaky_noisy_or2(self,features,attention_logit,is_training,reuse=False):
        pass
    def _attribute_gate_layer2(self,z,h,x,prob,reuse):
        pass

    def decision(self,c,h,reuse = False):
        with tf.variable_scope('decision_layer',reuse = reuse):
            w_c = tf.get_variable('w_c', [self.M, 1024], initializer=self.weight_initializer)
            #w_h = tf.get_variable('w_h', [self.H , 1024], initializer=self.weight_initializer)
            b_z = tf.get_variable('b_z',[1024],initializer=tf.constant_initializer(0.0))
            w = tf.get_variable('w', [1024,1], initializer=self.weight_initializer)
            b = tf.get_variable('b',[1],initializer=tf.constant_initializer(0.0))
        
            c = tf.stop_gradient(c)
            h = tf.stop_gradient(h)
            c = tf.nn.dropout(c,self.keep_prob)
            h = tf.nn.dropout(h,self.keep_prob)
            logit = tf.nn.relu(tf.matmul(c+h,w_c)+b_z)
            logit = tf.matmul(logit,w)+b
        return logit


    def build_model(self,features):
        #features = self.features
        #features = tf.reshape(features,[-1,64,2048])
        captions = self.captions
        decision_label = self.decision_label
        batch_size = tf.shape(features)[0]

        beta = 0.0
        c_context = 0.0
        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))

        # batch normalize feature vectors
        with tf.variable_scope('image_features'): 
            features_orig = slim.batch_norm(features, is_training = True)
        
        features_orig = tf.reshape(features_orig,[-1,self.K,2048])
        #project features
        features_proj,features_global = self._image_features(features_orig)
        c, h = self._get_initial_lstm(features=features_proj)
        x = self._word_embedding(inputs=captions_in) #(N,T,M)
        attribute_embedding = tf.expand_dims(self._word_embedding(self.attribute_list,reuse = True),0) #(1,1113,M)


        z,alpha = self._static_attribute_attention_layer(features)
  
        MIL_attention = self._MIL_attention(1.0,0.0,z,inital=True,reuse = False)
        prob,prob2 = self._MIL_Leaky_noisy_or(features,MIL_attention,is_training=True,reuse=False)
        #attention_logits= self._attribute_gate_layer(1.0,0.0,z,h,x[:,0,:],prob2,reuse=False)
        prob2 = tf.stop_gradient(prob2)
        attribute_vector,attribute_attention = self._attribute_vector_op(batch_size,prob2,attribute_embedding) #(N,1113)
   

        loss_gate = tf.constant(1,dtype=tf.float32)#self.loss_op(attention_logits,self.attributes_onehot,1,'gate')
        loss1 = self.loss_op(prob,self.attributes_onehot,1,'noisy_or')
        loss2 = self.loss_op(MIL_attention,self.attributes_onehot,1,'attention')
        loss_attribute =  loss1 + loss2

        loss = 0.0
        loss_first = 0.0
        loss_decision = 0.0
        
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,input_keep_prob=0.7,output_keep_prob=0.7)

        with tf.variable_scope('lstm', reuse=False):
            _, (c, h) = lstm_cell(inputs=tf.concat([features_global,attribute_vector],axis=1), state=[c, h])


        for t in range(self.T):
        
            x_new = tf.concat([x[:,t,:],attribute_vector],1)

            with tf.variable_scope('lstm', reuse=True):
              h_old = h
              c_old = c
              _, (c, h) = lstm_cell(inputs=x_new, state=[c_old, h_old])
            vs = self._visual_sentinel(h_old,c,x_new,reuse=(t!=0))

            h = tf.stop_gradient(h)
            vs = tf.stop_gradient(vs)

            c_context,c_hat_1,alpha,beta_1 = self._attention_layer(features_proj, features_orig,h, vs,reuse=(t!=0))
            logits_first = self._decode_layer(h, c_hat_1,stop_gradient = True,reuse=(t!=0))
            #c_context = tf.stop_gradient(c_context)
            #c_hat_1 = tf.stop_gradient(c_hat_1)
            #beta_1 = tf.stop_gradient(beta_1)
            

            gamma = self.decision(c_hat_1,h,reuse = (t!=0))
            loss_decision += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = gamma, labels = decision_label[:,t+1,None])* mask[:, t])
            
           
           
            MIL_attention = self._MIL_attention(beta_1,c_context,z,inital = False,reuse=True)
            _,prob2 = self._MIL_Leaky_noisy_or(features,MIL_attention,is_training=True,reuse=True)
            attention_logits= self._attribute_gate_layer(beta_1,c_context,z,h,x[:,t,:],prob2,reuse=(t!=0))
            attribute_vector2,attribute_attention2 = self._attribute_vector_op(batch_size,attention_logits,attribute_embedding)


            x_new = tf.concat([x[:,t,:],attribute_vector2],1)

            with tf.variable_scope('lstm', reuse=True):
              _, (c, h) = lstm_cell(inputs=x_new, state=[c_old, h_old])
            vs = self._visual_sentinel(h_old,c,x_new,reuse=True)
            _,c_hat, alpha,beta= self._attention_layer2(features_proj, features_orig,h,alpha,vs,reuse=(t!=0))
            #c_context,c_hat,alpha,beta = self._attention_layer(features_proj, features_orig,h, vs,reuse=True)
            #import ipdb; ipdb.set_trace()
            #c_hat = tf.stop_gradient(c_hat)
            #beta = tf.stop_gradient(beta)

            logits2 = self._inverse_word_emb(attribute_vector2,reuse=(t!=0))
            logits = self._decode_layer(h, c_hat,stop_gradient=False,reuse=True)
            loss_first += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_first, labels=captions_out[:, t]) * mask[:, t])
            loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=captions_out[:, t]) * mask[:, t])

        return attribute_attention,(loss+loss_first)/tf.reduce_sum(mask),loss_attribute,loss_gate,loss_decision/tf.reduce_sum(mask)

    def build_sampler(self,features,max_len=20):
        #features = self.features
        #features = tf.reshape(features,[-1,64,2048])
        batch_size = tf.shape(features)[0]
        # batch normalize feature vectors
        with tf.variable_scope('image_features'): 
            features_orig = slim.batch_norm(features, is_training = False)

        features_orig = tf.reshape(features_orig,[-1,self.K,2048])   
        features_proj,features_global = self._image_features(features_orig)
        c, h = self._get_initial_lstm(features=features_proj)

        alpha_list = []
        beta_list = []
        alpha2_list = []
        beta2_list = []
        value_list = []
        index_list = []
        gamma_list = []

        z,alpha = self._static_attribute_attention_layer(features)
        alpha_list.append(alpha)
        alpha2_list.append(alpha)

        x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
        attribute_embedding = tf.expand_dims(self._word_embedding(self.attribute_list,reuse = True),0) #(1,1113,M)
        MIL_attention = self._MIL_attention(1.0,0.0,z,inital = True,reuse = False)
        _,prob2 = self._MIL_Leaky_noisy_or(features,MIL_attention,is_training=False,reuse=False)
        #attention_logits= self._attribute_gate_layer(1.0,0.0,z,h,x,prob2,reuse=False)
        attribute_vector,attribute_attention = self._attribute_vector_op(batch_size,prob2,attribute_embedding)
        value,idex = tf.nn.top_k(attribute_attention,k=5) #(N,5)
        value_list.append(value)
        index_list.append(idex)

        sampled_word_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)


        with tf.variable_scope('lstm', reuse=False):
            _, (c, h) = lstm_cell(inputs=tf.concat([features_global,attribute_vector],axis=1), state=[c, h])
        
        

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)
                
            x_new = tf.concat([x,attribute_vector],1)

            with tf.variable_scope('lstm', reuse=True):
                h_old = h
                c_old = c
                _, (c, h) = lstm_cell(inputs=x_new, state=[c, h]) #(N,512)
            vs = self._visual_sentinel(h_old,c,x_new,reuse=(t!=0))
            

            c_context,c_hat_1, alpha,beta_1 = self._attention_layer(features_proj,features_orig ,h, vs,reuse=(t!=0))
            alpha_list.append(alpha)
            beta_list.append(beta_1)
            gamma = self.decision(c_hat_1,h,reuse = (t!=0)) #(N,1)
            gamma_list.append(tf.nn.sigmoid(gamma))
            
            MIL_attention = self._MIL_attention(beta_1,c_context,z,inital = False,reuse=True)
            _,prob2 = self._MIL_Leaky_noisy_or(features,MIL_attention,is_training=False,reuse=True)
            attention_logits= self._attribute_gate_layer(beta_1,c_context,z,h,x,prob2,reuse=(t!=0))
            attribute_vector2,attribute_attention2 = self._attribute_vector_op(batch_size,attention_logits,attribute_embedding)
            
            value,index = tf.nn.top_k(attribute_attention2,k=5) #(N,5)
            value_list.append(value)
            index_list.append(index)

            x_new = tf.concat([x,attribute_vector2],1)

            with tf.variable_scope('lstm', reuse=True):
              _, (c, h) = lstm_cell(inputs=x_new, state=[c_old, h_old])
            vs = self._visual_sentinel(h_old,c,x_new,reuse=True)
            _,c_hat, alpha,beta = self._attention_layer2(features_proj, features_orig,h,alpha, vs,reuse=(t!=0))
            #c_context,c_hat,alpha,beta = self._attention_layer(features_proj, features_orig,h+c_hat_1, vs,reuse=True)
            logits = self._decode_layer(h, c_hat ,stop_gradient = False,reuse=(t!=0))
            logits2 = self._inverse_word_emb(attribute_vector2,reuse=(t!=0))

            alpha2_list.append(alpha)
            beta2_list.append(beta)

            sampled_word = tf.argmax(logits, 1)
            sampled_word_list.append(sampled_word)

        gammas =  tf.transpose(tf.squeeze(gamma_list,2), (1, 0))  # (N, T)
        betas = tf.transpose(tf.squeeze(beta_list,2), (1, 0))  # (N, T)
        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T+1, K)
        betas2= tf.transpose(tf.squeeze(beta2_list,2), (1, 0))  # (N, T)
        alphas2 = tf.transpose(tf.stack(alpha2_list), (1, 0, 2))     # (N, T+1, K)
        values = tf.transpose(tf.stack(value_list), (1, 0, 2)) #(N,T+1,5)
        indices = tf.transpose(tf.stack(index_list), (1, 0, 2)) #(N,T+1,5)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        return gammas,alphas,betas, alphas2,betas2,values,indices, sampled_captions
