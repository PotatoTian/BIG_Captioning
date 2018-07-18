# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 09:58:33 2018

@author: junjiao
"""

import tensorflow as tf
import numpy as np
import time
import os
import pickle
import random
from nets import nets_factory
from preprocessing import inception_preprocessing#vgg_preprocessing
from preprocessing import image_processing
import numpy as np
import cv2
import random
from scipy import ndimage
from core.utils import *
from core.bleu import evaluate
from scipy import ndimage
import matplotlib
matplotlib.use('Agg') #use matplotlib without a display
import matplotlib.pyplot as plt
import skimage.transform
slim = tf.contrib.slim

class CaptioningSolver(object):
    def __init__(self, model, data, val_data, **kwargs):
        """
        NOTE: features are stored in batches of size 50 each
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17)
                - image_idxs: Indices for mapping caption to image of shape (400000, )
                - word_to_idx: Mapping dictionary from word to index
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path
            - model_path: String; model path for saving
            - test_model: String; model path for test
        """
        self.CNN_MODEL = 'inception_v3'#resnet_v1_152'
        self.model = model
        self.data = data
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 100)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.starter_learning_rate = kwargs.pop('learning_rate', 5e-4)
        self.starter_learning_rate_resnet = kwargs.pop('learning_rate_resnet', 5e-5)
        self.starter_learning_rate_classifier = kwargs.pop('learning_rate_classifier', 5e-4)
        self.print_bleu_every = kwargs.pop('print_bleu_every', 1000)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 5)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model',None)#'/data1/junjiaot/model/Resnet_Pretrain_adaptive_attribute34_new/model/2000-5')
        self.pretrained_resnet = kwargs.pop('pretrained_resnet','/data1/junjiaot/data/inception_v3.ckpt') #'/data1/junjiaot/data/inception_v3.ckpt'
        self.pretrained_classifier = kwargs.pop('pretrained_classifier',None)#'/data1/junjiaot/model/Resnet_Pretrain_adaptive_attribute34_new/classifier/2000-5')
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
        #self.num_feature_batch = 400
        #self.num_feature = self.num_feature_batch * 50
        self.image_save_dir = kwargs.pop('image_dir','/data1/junjiaot/model/Resnet_Pretrain_adaptive_attribute38/examples_')
        self.starting_epoch = kwargs.pop('starting_epoch', 0)
        self.image_size = kwargs.pop('image_size', 299)
        self.trainable_scopes_model = 'lstm,initial_lstm,word_embedding,image_features,attention_layer,the_second_attention_layer,Visual_Sentinel,decode_layer,attribute_attention_layer,decision_layer'
        self.trainable_scopes_classifier = 'MIL'#'MIL'
        self.checkpoint_exclude_scopes = 'InceptionV3/AuxLogits,InceptionV3/Logits,InceptionV3/Predictions,InceptionV3/PreLogits,' + self.trainable_scopes_model + ',' + self.trainable_scopes_classifier
        self.trainable_scopes_resnet = 'InceptionV3/Mixed_7c'#'resnet_v1_152/block4'
        self.is_finetuning_classifier = kwargs.pop('is_finetuning_classifier_classifier', True)
        self.is_finetuning_resenet = kwargs.pop('is_finetuning_resenet', False)
        self.finetune_classifier =  kwargs.pop('finetune_classifier', -1)
        self.finetune_resenet =  kwargs.pop('finetune_resenet', 20)
        self.attribute_list = np.array(load_pickle('/data1/junjiaot/data/train/multi_class_labels_list.pkl'))
        #self.lr = tf.placeholder(tf.float32)
        self.images = tf.placeholder(tf.float32, [None,self.image_size,self.image_size,3],'images')
        self.image = tf.placeholder(tf.uint8,[None,None,3],'image')
      

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def _get_variables_to_train(self,trainable_scopes):
        scopes = [scope.strip() for scope in trainable_scopes.split(',')]
        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        return variables_to_train

    def _get_variables_to_save(self,trainable_scopes):
        scopes = [scope.strip() for scope in trainable_scopes.split(',')]
        variables_to_save = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_save.extend(variables)
            for variable in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES,scope):
                if variable not in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope):
                    variables_to_save.append(variable)
        return variables_to_save

    def get_init_fn(self):
        """Returns a function run by the chief worker to warm-start the training."""
        #with G.as_default():
        exclusions = [scope.strip() for scope in self.checkpoint_exclude_scopes.split(',')]
        variables_to_restore = []
        for var in slim.get_model_variables():
        #for var in tf.trainable_variables(scope ='InceptionV3' ):
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
        return variables_to_restore
            

    def test(self,images):
        '''
        Under construction
        images: (N,hight,width,3)
        '''
        _,_,betas, alphas2,betas2,values,indices,generated_captions = self.model_build(self.images,is_training=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

        model_variables_to_train = self._get_variables_to_train(self.trainable_scopes_model)
        saver_model = tf.train.Saver(model_variables_to_train)
        if self.pretrained_model is not None:
            print ("Start training with pretrained Model..")
        selfaver_model.restore(sess, self.pretrained_model)
        #Attribute classifier saver
        classifier_variables_to_train = self._get_variables_to_train(self.trainable_scopes_classifier)
        saver_classifier = tf.train.Saver(classifier_variables_to_train)
        if self.pretrained_classifier is not None:
            print('Start with pretrained classifier')
            saver_classifier.restore(sess,self.pretrained_classifier)
        #resnet saver
        with tf.get_default_graph().as_default():
            variables_to_restore = self.get_init_fn()
        saver_resnet = tf.train.Saver(variables_to_restore,max_to_keep = 1)
        saver_resnet.restore(sess,self.pretrained_resnet)
        if self.pretrained_model is not None:
            print ("Start training with pretrained Model..")
            saver.restore(sess, self.pretrained_model)

        gamma,alpha,beta,alpha2,beta2,value,index,gen_caps = sess.run([gammas,alphas,betas, 
                                            alphas2,betas2,
                                            values,indices,
                                            generated_captions], feed_dict)

        decoded = decode_captions(gen_caps, self.model.idx_to_word) #gen_caps:(N,T)
        print ("Generated caption: %s\n" %decoded)
        index = np.reshape(index,[-1])
        attributes = self.attribute_list[index]
        prediction = decode_captions(np.reshape(attributes,[-1,5]), self.model.idx_to_word)  # index:(N,T+1,5)

        image = cv2.imread(file_batch[0])
        b,g,r = cv2.split(image)
        rgb_img = cv2.merge([r,g,b])
        self.visualization(e,'test',image_batch[0][:,:,:]/2+0.5,decoded[0],gamma,
                            beta,alpha,beta2,alpha2,
                            value,prediction)





    def model_build(self,images,is_training=True):
        global_step = tf.train.get_or_create_global_step()
        with tf.device('/gpu:1'):
          network_fn = nets_factory.get_network_fn(
                      self.CNN_MODEL,
                      num_classes=1001,
                      weight_decay=0.0001,
                      is_training=is_training)
        logits,end_points = network_fn(images)
        fc7 = end_points['Mixed_7c']#['resnet_v1_152/block4']##end_points['resnet_v2_101/block4/unit_3/bottleneck_v2/conv3']
        #print(fc7.shape)
        # build graphs for training model and sampling captions
        if is_training:
            attri_atten,loss,loss_attribute,loss_gate,loss_decision = self.model.build_model(fc7)
            print('build success')
            return attri_atten,loss,loss_attribute,loss_gate,loss_decision
        else:
            gammas,alphas,betas, alphas2,betas2,values,indices, generated_captions = self.model.build_sampler(fc7,max_len=20)
            print('build sampler success')
            return gammas,alphas,betas, alphas2,betas2,values,indices,generated_captions

    def loss_op(self,loss,learning_rate,learning_rate_resnet,learning_rate_classifier):
        ##Specify trainable variables for the optimizer
        model_variables_to_train = self._get_variables_to_train(self.trainable_scopes_model)
        classifier_variables_to_train = self._get_variables_to_train(self.trainable_scopes_classifier)
        resnet_variables_to_train = []
        #import ipdb; ipdb.set_trace()
        if self.is_finetuning_resenet == True:
             resnet_variables_to_train = self._get_variables_to_train(self.trainable_scopes_resnet)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads = tf.gradients(loss, model_variables_to_train + classifier_variables_to_train + resnet_variables_to_train)
            #import ipdb;ipdb.set_trace()
            clipped_grads = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in zip(grads, model_variables_to_train+classifier_variables_to_train+resnet_variables_to_train)]
            with tf.name_scope('adam_optimizer'):
            
                clipped_model_grads = clipped_grads[:len(model_variables_to_train)]
                op_model = tf.train.AdamOptimizer(learning_rate,beta1=0.8)
                train_op_model = op_model.apply_gradients(clipped_model_grads)

                clipped_classifier_grads = clipped_grads[len(model_variables_to_train):len(model_variables_to_train+classifier_variables_to_train)]
                op_classifier = tf.train.AdamOptimizer(learning_rate_classifier,beta1=0.8)
                train_op_classifier = op_classifier.apply_gradients(clipped_classifier_grads)
                if self.is_finetuning_resenet == True:
                  clipped_resnet_grads = clipped_grads[len(model_variables_to_train+classifier_variables_to_train):]
                  op_resnet = tf.train.AdamOptimizer(learning_rate_resnet,beta1=0.8)
                  train_op_resenet = op_resnet.apply_gradients(clipped_resnet_grads)

        if self.is_finetuning_resenet == True:
          return train_op_model,train_op_classifier,train_op_resenet,grads
        else:
          return train_op_model,train_op_classifier,tf.constant(0),grads

    def _print_trainable(self):
        model_variables_to_train = self._get_variables_to_train(self.trainable_scopes_model)
        print('model trainable variables:')
        for var in model_variables_to_train:
            print (var.name)
        classifier_variables_to_train = self._get_variables_to_train(self.trainable_scopes_classifier)
        print('classifier trainable variables:')
        for var in classifier_variables_to_train:
            print (var.name)
        if self.is_finetuning_resenet == True:
          resnet_variables_to_train = self._get_variables_to_train(self.trainable_scopes_resnet)
          print('resnet trainable variables:')
          for var in resnet_variables_to_train:
              print(var.name)

    def preprocess_image(self,image,is_training = True):
        return image_processing.process_image(image,is_training=is_training, height = self.image_size, width =self.image_size )

    
    def visualization(self,e,file_name,image,decoded,gammas,betas,alphas,betas2,alphas2,values,predictions):
        '''
        gammas: [1,T]
        betas: [1,T]
        alphas: [1,T+1,K]
        values: [1,T+1,5]
        predicitons: [T+1,1]
        '''
        # Plot original image
        save_dir = self.image_save_dir + str(e) + '/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        alpha_list = np.array([alphas,alphas2])#[2,1,T+1,k]
        beta_list = [betas,betas2] #[2,1,T+1]
        words = decoded.split(" ")
        modes = ['_first_attention','_refined_attention']
        for i,mode in enumerate(modes):
            fig = plt.figure(i,dpi = 500)
            ax = fig.add_subplot(11, 4, 1)
            ax.imshow(image, aspect='equal',extent = (-8,8,-8,8))
            ax.text(8.5, 0,file_name.split('/')[5].split('.')[0], color='black', fontsize=3)
            ax.axis('off')
            ax = fig.add_subplot(11, 4, 2)
            ax.imshow(image, aspect='equal',extent = (-8,8,-8,8))
            alp_curr = alpha_list[i][0,0].reshape(8,8)
            alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=37.375,sigma=20)
            ax.imshow(alp_img,alpha = 0.5, aspect='equal', extent = (-8,8,-8,8))
            ax.axis('off')
            ax = fig.add_subplot(11,4,3)
            for j,predict in enumerate(predictions[0].split(' ')):
                ax.text(0,(0.2*(4-j)),'%s(%.2f)'%(predict,values[0,0,j]),color='black', fontsize=3)
            ax.axis('off')
            # Plot images with first attention weights
            for t in range(len(words)): 
                if t > 18:
                    break
                ax = fig.add_subplot(11, 4, 6+2*t-1)
                ax.text(8.5, 0, '%s(%.2f,%.2f)'%(words[t],1-beta_list[i][0,t],gammas[0,t]) , color='black', fontsize=3)
                ax.imshow(image, aspect='equal',extent = (-8,8,-8,8))
                alp_curr = alpha_list[i][0,t+1].reshape(8,8)
                alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=37.375,sigma=20)
                ax.imshow(alp_img,alpha = 0.5, aspect='equal',extent = (-8,8,-8,8))
                ax.axis('off')
                ax = fig.add_subplot(11, 4, 6+2*t)
                for j,predict in enumerate(predictions[t+1].split(' ')):
                    ax.text(0,(0.2*(4-j)),'%s(%.2f)'%(predict,values[0,t+1,j]),color='black', fontsize=3)
                ax.axis('off')
            #import ipdb;ipdb.set_trace()
            save_path = save_dir + file_name.split('/')[5].split('.')[0] + mode+'.jpg'
            plt.savefig(save_path) 
            print('Example saved at ' + save_path)   
        plt.close('all')
    def train(self):

        # train/val dataset
        n_examples = self.data['captions'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size)) #3124
        captions = self.data['captions']
        image_idxs = self.data['image_idxs']
        annotations = load_pickle('/data1/junjiaot/data/train/train.annotations.pkl')
        file_names = annotations['file_name']
        
        #coco_attributes = load_pickle('/home/junjiaot/data/cocodata/data/train/train.multi_class_labels_abs.pkl') #(82783,30)
        coco_attributes_onehot = load_pickle('/data1/junjiaot/data/train/train.multi_class_labels.pkl') #(82783,1113)
        print('number of files: {}, number of captions: {}'.format(len(file_names),len(captions)))
        #import ipdb;ipdb.set_trace()
        people_label = load_pickle('/data1/junjiaot/data/train/train.people_label.pkl')

        #build model
        #batch_train = tf.split(axis=0, num_or_size_splits=self.batch_size, value=self.images)
        #for i in range(self.batch_size):
        #   batch_train[i] = tf.expand_dims(vgg_preprocessing.preprocess_image(tf.squeeze(batch_train[i]),224,224,True),0)
        #image_train =  tf.concat(axis=0, values=batch_train)

        #batch_val = tf.split(axis=0, num_or_size_splits=self.batch_size, value=self.images)
        #for i in range(self.batch_size):
        #   batch_val[i] = tf.expand_dims(vgg_preprocessing.preprocess_image(tf.squeeze(batch_val[i]),224,224,False),0)
        #image_val =  tf.concat(axis=0, values=batch_val)

        image = tf.image.convert_image_dtype(self.image, dtype=tf.float32)
        image_train = self.preprocess_image(image,is_training = True)
        image_val = self.preprocess_image(image,is_training = False)

        attri_atten,loss,loss_attribute,loss_gate,loss_decision = self.model_build(self.images,is_training=True)
        with tf.variable_scope(tf.get_variable_scope()):
            tf.get_variable_scope().reuse_variables()
            gammas,alphas,betas, alphas2,betas2,values,indices,generated_captions = self.model_build(self.images,is_training=False)

       
        

        # learning rate decay
        global_step_model = tf.get_variable('lr_rate',initializer = tf.constant(0),trainable= False)
        global_step_classifier = tf.get_variable('lr_rate_classifier',initializer = tf.constant(0),trainable= False)
        global_step_resnet = tf.get_variable('lr_rate_resnet',initializer = tf.constant(0),trainable= False)

        increment_global_step_model = tf.assign(global_step_model,global_step_model+1)
        increment_global_step_classifier = tf.assign(global_step_classifier,global_step_classifier+1)
        increment_global_step_resenet = tf.assign(global_step_resnet,global_step_resnet+1)

        learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step_model,
                                           50, 0.5, staircase=False)
        learning_rate_resnet = tf.train.exponential_decay(self.starter_learning_rate_resnet, global_step_resnet,
                                           10, 0.5, staircase=False)
        learning_rate_classifier = tf.train.exponential_decay(self.starter_learning_rate_classifier, global_step_classifier,
                                           10, 0.5, staircase=False)
        #train op
        train_op_model,train_op_classifier,train_op_resenet,grads = self.loss_op(loss+loss_attribute+loss_gate+loss_decision,learning_rate,learning_rate_resnet,learning_rate_classifier)

        # summary op
        #tf.summary.scalar('batch_loss', loss)
        #for var in tf.trainable_variables():
        #    tf.summary.histogram(var.op.name, var)
        #for grad, var in grads_and_vars:
        #    tf.summary.histogram(var.op.name+'/gradient', grad)

        #ßsummary_op = tf.summary.merge_all()
        print('Learning rate:{}'.format(self.starter_learning_rate))
        print ("The number of epoch: %d" %self.n_epochs)
        print ("Data size: %d" %n_examples)
        print ("Batch size: %d" %self.batch_size)
        print ("Iterations per epoch: %d" %n_iters_per_epoch)

        self._print_trainable()
        #import ipdb;ipdb.set_trace()

        config = tf.ConfigProto(allow_soft_placement = True)
        #config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True




        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer =tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            #Lauguage model saver
            model_variables_to_train = self._get_variables_to_save(self.trainable_scopes_model)
            saver_model = tf.train.Saver(model_variables_to_train)
            if self.pretrained_model is not None:
                print ("Start training with pretrained Model..")
                saver_model.restore(sess, self.pretrained_model)
            #Attribute classifier saver
            classifier_variables_to_train = self._get_variables_to_save(self.trainable_scopes_classifier)
            saver_classifier = tf.train.Saver(classifier_variables_to_train)
            if self.pretrained_classifier is not None:
                print('Start with pretrained classifier')
                saver_classifier.restore(sess,self.pretrained_classifier)
            #resnet saver
            with tf.get_default_graph().as_default():
              variables_to_restore = self.get_init_fn()
            saver_resnet = tf.train.Saver(variables_to_restore,max_to_keep = 1)
            saver_resnet.restore(sess,self.pretrained_resnet)

            import ipdb;ipdb.set_trace()
            prev_loss = -1
            curr_loss = 0


            image_dir = '/home/junjiaot/data_local/train2014'#/data1/junjiaot/image/resized_224_aspect/train2014'
            image_files = []
            for file in file_names:
                image_files.append(os.path.join(image_dir,file.split('/')[2]))
            image_files = np.array(image_files)
                
            #self._validation_score(0,sess,generated_captions,starter_learning_rate)
            for e in range(self.starting_epoch,self.n_epochs):
                # start learning rate decay
                if e>20:
                    sess.run(increment_global_step_model)
                if e>self.finetune_classifier and self.is_finetuning_classifier:
                    sess.run(increment_global_step_classifier)
                if e>self.finetune_resenet and self.is_finetuning_resenet:
                    sess.run(increment_global_step_resenet)

                print('epoch:{}, learning_rate:{}'.format(e+1,sess.run(learning_rate)))
                start_t = time.time()

                temp = list(zip(image_files,captions,people_label,image_idxs))
                random.shuffle(temp)
                image_files,captions,people_label,image_idxs = zip(*temp)
                captions = np.array(captions)#change to numpy array to deal with out of bound index
                people_label = np.array(people_label)
                #pickle_files = []
                #pkl_dir = '/home/junjiaot/data_local/train/resnet152_v1_feature/'#'/data1/junjiaot/data/train/resnet152_v1_feature/'
                #for file in file_names:
                #pickle_files = np.array(pickle_files)
                


                start = 0
                end = self.batch_size
                for i in range(n_iters_per_epoch):
                    file_batch = image_files[start:end]
                    captions_batch = captions[start:end]
                    idx_batch = image_idxs[start:end]
                    people_label_batch = people_label[start:end]
                    #attribute_batch = []
                    attribute_onehot_batch = []
                    #for idx in idx_batch:
                      #attribute_batch.append([idx])
                      #attribute_onehot_batch.append(coco_attributes_onehot[idx])
                    attribute_onehot_batch = coco_attributes_onehot[idx_batch,:]
                    #features_batch = np.array(list(map(lambda x:pickle.load(open(x,'rb')),file_batch))).squeeze()
                    image_batch = []
                    for image_file in file_batch:
                        image = cv2.imread(image_file)
                        b,g,r = cv2.split(image)
                        rgb_img = cv2.merge([r,g,b])
                        image_batch.append(sess.run(image_train,feed_dict={self.image:rgb_img}))

                    image_batch = np.array(image_batch)
                    if image_batch.ndim == 3:
                            image_batch = image_batch[np.newaxis,:,:,:]
                    
                    feed_dict = {self.images: image_batch, self.model.captions: captions_batch,
                                 self.model.decision_label: people_label_batch,
                                 #self.model.attributes:attribute_batch,
                                 self.model.attributes_onehot: attribute_onehot_batch,
                                 self.model.keep_prob:0.5}
                    #import ipdb; ipdb.set_trace()
                    if e>self.finetune_classifier and self.is_finetuning_classifier:
                        _,_,atten,l,l_attribute ,l_gate,l_decision= sess.run([train_op_model,train_op_classifier,attri_atten,loss,loss_attribute,loss_gate,loss_decision], feed_dict)
                    elif e>self.finetune_resenet and self.is_finetuning_resenet:
                        _,_,_,atten,l,l_attribute,l_gate,l_decision = sess.run([train_op_model,train_op_classifier,train_op_resenet,beta,attri_atten,loss,loss_attribute,loss_gate,loss_decision], feed_dict)
                    else:
                        _,atten,l,l_attribute,l_gate,l_decision = sess.run([train_op_model,attri_atten,loss,loss_attribute,loss_gate,loss_decision], feed_dict)
                    
                    curr_loss += l
                                        # write summary for tensorboard visualization
                    if i % 10 == 0:
                        print("Epoch:",e," Iteration:",i," Loss:",l, 'Attribute_loss:',l_attribute,'Gate_loss',l_gate, 'Decision loss:',l_decision)
                        #zipped = zip(atten[0,:])
                        #atten = zip(*sorted(zipped, key=lambda x: x[0],reverse=True))
                        print(sorted(atten[0,:],reverse = True)[:10])
                        #print(s_gate[:10])
                        #print(beta_list[0,:])

                        #summary = sess.run(summary_op, feed_dict)
                        #summary_writer.add_summary(summary, e*n_iters_per_epoch + i)
                    

                    if i % self.print_every == 0:
                        #print ("\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f :%.5f" %(e+1, i+1, l,l_attribute))
                        ground_truths = captions[image_idxs == idx_batch[0]]
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print ("Ground truth %d: %s" %(j+1, gt))
                        feed_dict = {self.images: image_batch[0][np.newaxis,:,:,:],        
                                     #self.model.attributes:attribute_batch[0][np.newaxis,:],
                                     self.model.attributes_onehot:attribute_onehot_batch[0][np.newaxis,:],self.model.keep_prob:1.0}
                        gamma,alpha,beta,alpha2,beta2,value,index,gen_caps = sess.run([gammas,alphas,betas, 
                                                    alphas2,betas2,
                                                    values,indices,
                                                    generated_captions], feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word) #gen_caps:(N,T)
                        print ("Generated caption: %s\n" %decoded)
                        index = np.reshape(index,[-1])
                        attributes = self.attribute_list[index]
                        prediction = decode_captions(np.reshape(attributes,[-1,5]), self.model.idx_to_word)  # index:(N,T+1,5)

                        image = cv2.imread(file_batch[0])
                        b,g,r = cv2.split(image)
                        rgb_img = cv2.merge([r,g,b])
                        self.visualization(e,file_batch[0],image_batch[0][:,:,:]/2+0.5,decoded[0],gamma,
                                            beta,alpha,beta2,alpha2,
                                            value,prediction)
                        

                     # print out BLEU scores and file write
                    if i % 1000 == 0:
                        self._validation_score(e,sess,image_val,generated_captions,sess.run(learning_rate))

                    # save model's parameters
                    if i % 1000 == 0:
                        saver_model.save(sess, os.path.join(self.model_path, 'model',str(i)), global_step=e)
                        saver_classifier.save(sess,os.path.join(self.model_path, 'classifier',str(i)),global_step=e)
                        saver_resnet.save(sess,os.path.join(self.model_path, 'resenet',str(i)),global_step=e)
                        print ("model-%s saved." %(e))
                    start = end
                    end = end + self.batch_size
                    #if end > len(file_names):
                    #    end = len(file_names) - 1
                    #    start = end - self.batch_size


                print ("Previous epoch loss: ", prev_loss)
                print ("Current epoch loss: ", curr_loss)
                print ("Elapsed time: ", time.time() - start_t)
                prev_loss = curr_loss
                curr_loss = 0




    def _validation_score(self,e,sess,image_val,generated_captions,lr):
        file_names = load_pickle('/data1/junjiaot/data/val/val.file.names.pkl')
        print('number of validation features:', len(file_names))
        coco_attributes = load_pickle('/data1/junjiaot/data/val/val.multi_class_labels_abs.pkl')
        #print(coco_attributes.shape)
        coco_attributes_onehot = load_pickle('/data1/junjiaot/data/val/val.multi_class_labels.pkl') #(82783,1113)
        #pkl_dir = '/home/junjiaot/data_local/val/resnet152_v1_feature/'#'/data1/junjiaot/data/val/resnet152_v1_feature/'
        #pickle_files = []
        #for file in file_names:
        #    pickle_files.append(os.path.join(pkl_dir,file.split('/')[2].split('.')[0]+'.pkl'))
        #pickle_files = np.array(pickle_files)
        image_dir = '/home/junjiaot/data_local/val2014'#'/data1/junjiaot/image/resized_224_aspect/val2014'#'/data1/junjiaot/image/resized_224_aspect/train2014'
        image_files = []
        for file in file_names:
            image_files.append(os.path.join(image_dir,file.split('/')[2]))
        image_files = np.array(image_files)

        start = 0
        end = self.batch_size
        n_iters_val = int(np.ceil(float(len(file_names))/self.batch_size))
        all_gen_cap = np.ndarray((len(file_names), 20))
        for i in range(n_iters_val):
            #features_batch = np.array(list(map(lambda x:pickle.load(open(x,'rb')),pickle_files[start:end]))).squeeze()
            image_batch = []
            for image_file in image_files[start:end]:
                image = cv2.imread(image_file)
                b,g,r = cv2.split(image)
                rgb_img = cv2.merge([r,g,b])
                image_batch.append(sess.run(image_val,feed_dict={self.image:rgb_img}))
            #image_batch = image_val_all[start:end,:,:,:]
            #attribute_batch = coco_attributes[start:end]
            attribute_onehot_batch = coco_attributes_onehot[start:end]
            feed_dict = {self.images: image_batch,#self.model.attributes:attribute_batch,
                         self.model.attributes_onehot: attribute_onehot_batch,
                         self.model.keep_prob:1.0}
            gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
            #import ipdb; ipdb.set_trace()
            all_gen_cap[i*self.batch_size:(i+1)*self.batch_size] = gen_cap[:]
            start = end
            end = end + self.batch_size
            #if end > len(file_names):
            #    residual = start - (len(file_names) - self.batch_size)
            #    end = len(file_names) - 1
            #    start = end - self.batch_size

        all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
        save_pickle(all_decoded, "/data1/junjiaot/data/val/val.candidate.captions.pkl")
        print('Calculating scores...')
        scores = evaluate(data_path='/data1/junjiaot/data', split='val', get_scores=True)
        write_bleu(scores=scores, path=os.path.join(self.model_path,'Score'), epoch=e,lr=lr)














