!pip install git+https://github.com/keras-team/keras-nlp.git  -q

import os

os.environ["KERAS_BACKEND"] = "tensorflow"


import time

import keras_nlp
import tensorflow as tf
import tensorflow_datasets as tfds

import keras_core as keras

BATCH_SIZE = 1
NUM_BATCHES = 600
EPOCHS = 4  # Can be set to a higher value for better results
MAX_ENCODER_SEQUENCE_LENGTH = 1000
MAX_DECODER_SEQUENCE_LENGTH = 1000
MAX_GENERATION_LENGTH = 40

!pip install datasets
!pip install tensorflow-datasets

from datasets import load_dataset
data= load_dataset("gopalkalpande/bbc-news-summary",split="train",trust_remote_code=True)

import tensorflow as tf
a=[]
b=[]
for i in range(1000):
  a.append(data[i]['Articles'])
  b.append(data[i]['Summaries'])
# Assuming your data is loaded into separate NumPy arrays for features and labels
features = {'encoder_text':a,'decoder_text':b}  # Your test data features as a NumPy array

dataset = tf.data.Dataset.from_tensor_slices(features)
dataset = dataset.batch(BATCH_SIZE)
print(dataset)    

import requests

def get_data_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print("Error:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None

# Replace 'your_url_here' with the actual URL you want to retrieve data from
url = 'https://datasets-server.huggingface.co/rows?dataset=ccdv%2Fpubmed-summarization&config=document&split=train&offset=0&length=100'

data = get_data_from_url(url)

a=[]
b=[]

for i in range(100):
  a.append(data['rows'][i]['row']['article'])
  b.append(data['rows'][i]['row']['abstract'].replace('\n',''))

import tensorflow as tf

# Assuming your data is loaded into separate NumPy arrays for features and labels
features = {'encoder_text':a,'decoder_text':b}  # Your test data features as a NumPy array

dataset = tf.data.Dataset.from_tensor_slices(features)
dataset = dataset.batch(BATCH_SIZE)
print(dataset)

preprocessor = keras_nlp.models.BartSeq2SeqLMPreprocessor.from_preset(
    "bart_base_en",
    encoder_sequence_length=MAX_ENCODER_SEQUENCE_LENGTH,
    decoder_sequence_length=MAX_DECODER_SEQUENCE_LENGTH,
)

bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset(
    "bart_base_en",preprocessor=preprocessor
)

bart_lm.summary()

import keras_nlp
from keras_nlp.models import BartSeq2SeqLM
from keras.losses import SparseCategoricalCrossentropy
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
    epsilon=1e-6,
    global_clipnorm=1.0,  # Gradient clipping.
    name="adamw"
)
# Exclude layernorm and bias terms from weight decay.
optimizer.exclude_from_weight_decay(var_names=["bias"])
optimizer.exclude_from_weight_decay(var_names=["gamma"])
optimizer.exclude_from_weight_decay(var_names=["beta"])

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bart_lm.compile(
    optimizer=optimizer,
    loss=loss,
    weighted_metrics=["accuracy"],
)

bart_lm.fit(dataset, epochs=3)

test_data= load_dataset("gopalkalpande/bbc-news-summary",split="train",trust_remote_code=True)

def generate_text(model, input_text, max_length=500, print_time_taken=False):
    # start = time.time()
    output = model.generate(input_text, max_length=max_length)
    # end = time.time()
    # print(f"Total Time Elapsed: {end - start:.2f}s")
    return output

document=data[1001]['Articles']

Provided_Summary=test_data[1001]['Summaries']

# Generate summaries.
generated_summaries = generate_text(
    bart_lm,
f""" {document} """,
print_time_taken=True,
)
print("Main text:    "+document)
print()
print("Provided Summary:   "+Provided_Summary)
print() 
print("Model generated summary: "+generated_summaries)

