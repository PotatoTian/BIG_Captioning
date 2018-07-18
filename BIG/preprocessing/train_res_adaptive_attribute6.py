# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:12:03 2018

@author: junji
"""

from core.res_solver_adaptive_attribute6 import CaptioningSolver
from core.res_model_adaptive_attribute6 import CaptionGenerator
from core.utils import load_coco_data



def main():
    # load train dataset
    data = load_coco_data(data_path='/data1/junjiaot/data', split='train')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path='/data1/junjiaot/data', split='val')
    #print(data['file_names'].shape,data['captions'].shape,data['image_idxs'].shape)
    #print(max(data['image_idxs']))
    #model/adaptive_attention_REINFORCE/
    #'model/adaptive_attention/3_26_2018/model-13
    model = CaptionGenerator(word_to_idx, dim_feature=[49, 2048], dim_embed=512,
                                       dim_hidden=512, n_time_step=16,alpha_c=1.0, dropout=True)

    solver = CaptioningSolver(model, data, val_data, n_epochs=50, batch_size=128, update_rule='adam',
                                          learning_rate=5e-4, print_bleu_every= 1000, save_every=1000, image_path='./image/',
                                    pretrained_model=None, model_path='model/Resnet_Pretrain_adaptive_attribute6/', test_model='model/lstm/model-5',
                                     print_bleu=True, log_path='log/')

    solver.train()

if __name__ == "__main__":
    main()
