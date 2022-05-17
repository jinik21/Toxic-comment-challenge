from flask import Flask, request, url_for,render_template,jsonify
import os
import random
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from transformers import TFAutoModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import transformers
# import json
AUTOTUNE = tf.data.experimental.AUTOTUNE
# UPLOAD_FOLDER = './static/uploads'
config = {
    'seed' : 42,
    'model': 'jplu/tf-xlm-roberta-base',
    'group': 'xlmbert',
    
    'batch_size': 16,
    'max_length': 64,
    
    'device' : 'GPU',
    'epochs' : 7,

    'test_size' : 0.1,
    'lr': 5e-6,
    'use_transfer_learning' : False,
    
    'use_wandb': False,
    'wandb_mode' : 'online',
}
print(transformers.__version__)
class ChaiiDataset:
    def __init__(self, max_length,  tokenizer):
        self.max_length = max_length
        self.pad_on_right = tokenizer.padding_side == "right"
        self.tokenizer = tokenizer
    def run_tokenizer( self, data):
        tokenized_data = self.tokenizer(
            list(data['commentText'].values),
            max_length = self.max_length,
            padding='max_length',
            truncation=True
        )
        return tokenized_data
    def prepare_tf_data_pipeline(self, data, batch_size = 16, type='train'):
        tokenized_data = self.run_tokenizer(data)
        print("tokenization done")
        def map_func(tokenized_data, label):
            input_ids = tokenized_data['input_ids']
            token_type_ids = 0
            attention_mask = tokenized_data['attention_mask']
            return {'input_ids':input_ids, 
                    'token_type_ids':token_type_ids, 
                    'attention_mask': attention_mask}, \
                    label
        def map_func_eval(tokenized_data):
            input_ids = tokenized_data['input_ids']
            token_type_ids = 0
            attention_mask = tokenized_data['attention_mask']
            return {'input_ids':input_ids, 
                    'token_type_ids':token_type_ids, 
                    'attention_mask': attention_mask}
        if type=='train':
            dataset_train_raw = tf.data.Dataset.from_tensor_slices((tokenized_data, data['label']))
            dataset_train = dataset_train_raw.map(map_func) \
                            .shuffle(10) \
                            .batch(batch_size, drop_remainder=True) \
                            .prefetch(buffer_size=AUTOTUNE) 
            return dataset_train
        if type =='valid':
            dataset_valid_raw = tf.data.Dataset.from_tensor_slices((tokenized_data, data['label']))
            dataset_valid = dataset_valid_raw.map(map_func) \
                            .batch(batch_size)
            return dataset_valid            
        if type == 'eval':
            dataset_eval_raw = tf.data.Dataset.from_tensor_slices((tokenized_data))
            dataset_eval = dataset_eval_raw.map(map_func_eval) \
                            .batch(batch_size)
                
            return dataset_eval    
tokenizer = AutoTokenizer.from_pretrained(config['model'])
print(tokenizer)
dataset = ChaiiDataset(config['max_length'],  tokenizer)   

from transformers import AutoModel
def get_keras_model():
    pretrained_model =  TFAutoModel.from_pretrained(config['model'])
    
    input_ids = layers.Input(shape=(config['max_length'],),
                             name='input_ids', 
                             dtype=tf.int32)
    token_type_ids = layers.Input(shape=(config['max_length'],),
                                  name='token_type_ids', 
                                  dtype=tf.int32)
    attention_mask = layers.Input(shape=(config['max_length'],),
                                  name='attention_mask', 
                                  dtype=tf.int32)
    embedding = pretrained_model(input_ids, 
                     token_type_ids=token_type_ids, 
                     attention_mask=attention_mask)[0]
    x1 = tf.keras.layers.Dropout(0.1)(embedding) 
    x1 = tf.keras.layers.Conv1D(1,1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dense(1, activation='sigmoid')(x1)
    model = keras.Model(inputs=[input_ids, 
                                token_type_ids, 
                                attention_mask],
                        outputs=x1)
    
    return model
model = get_keras_model()
model.load_weights('./static/best_model.h5')

app = Flask(__name__)
app.secret_key="12345678"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///db.sqlite3'#DBURI
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

admin="admin"
passwordc="kuchbhi@1234"
@app.route('/data',methods=['GET','POST'])
def data():
    if request.method == "POST":
        try:
            json_data=request.json
            r=json_data["comment"]
            df_test=pd.dataframe(r)
            test_dataset = dataset.prepare_tf_data_pipeline(df_test, type='eval')
            preds = model.predict(test_dataset, verbose = 1, workers=4)
            pds= preds>0.5
            ans=str(pds[0])
            return jsonify({"data":ans})
        except:
            return jsonify({"data":"Error While processing"})
    return jsonify({"data":"Invalid Request"})

if __name__ == "__main__":
    app.run()