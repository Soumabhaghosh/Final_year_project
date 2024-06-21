!pip install Pillow==9.0.0

!pip install pytesseract transformers datasets rouge-score nltk tensorboard py7zr --upgrade


dataset_id = "cnn_dailymail"
from datasets import load_dataset
from datasets import Dataset,DatasetDict

# Load dataset from the hub
dataset = load_dataset(dataset_id,"3.0.0")
# dataset=dataset.take(1000)
print(f"Train dataset size: {len(dataset['train'].take(1000))}")
print(f"Test dataset size: {len(dataset['test'].take(100))}")
dataset = DatasetDict({
    'train': dataset['train'].take(1000),
    'test': dataset['test'].take(100)
})
print(dataset)

dataset['train'].take(1000)

!pip install datasets
!pip install tensorflow-datasets

import tensorflow as tf
import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import load_dataset
# from tensorflow.datasets import load


# Load the dataset
dataset = load_dataset("gigaword")
print(dataset)

!pip install pandas


import pandas as pd
from sklearn.model_selection import train_test_split

# Attempt to read the file with 'latin1' encoding
df = pd.read_csv('/kaggle/input/news-summary/news_summary.csv', encoding='latin1')

# Display the first few rows of the dataframe
# df
df= df.dropna(subset=['ctext', 'text'])
df

import pandas as pd
from datasets import Dataset,DatasetDict

# Load the CSV file using pandas
df = pd.read_csv('/kaggle/input/news-summary/news_summary.csv', encoding='latin1')

# Display the first few rows of the dataframe
# print(df.head())

# Ensure the dataframe has the necessary columns: 'document' and 'summary'
# Rename columns if necessary (adjust column names as needed)
df = df.rename(columns={'article': 'ctext', 'abstract': 'text'})

# train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)
# validation_df, test_df = train_test_split(temp_df, test_size=0.01, random_state=42)

# Convert the pandas DataFrame to a Hugging Face Dataset
train_dataset = Dataset.from_pandas(df[['ctext', 'text']])
# test_dataset = Dataset.from_pandas(test_df[['ctext', 'text']])
# train_dataset = train_dataset.drop_index()

filtered_data = []

# # Loop through each row in the dataset
for i in range(len(train_dataset)):
    row = train_dataset[i]

    # Check if either 'ctext' or 'text' contains '<class 'NoneType'>'
    if isinstance(row['ctext'],str)==True and isinstance(row['text'],str)==True :
        filtered_data.append(row)

# Create a new dataset with the filtered data
filtered_dataset = Dataset.from_list(filtered_data, features=train_dataset.features)




dataset = DatasetDict({
    'train': filtered_dataset,
    'test': filtered_dataset
})


# # Display the dataset
print(dataset)

# Verify the number of rows and structure of the dataset
# print(f"Number of rows: {len(dataset)}")
# print(dataset.features)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id="google/flan-t5-base"

# Load tokenizer of FLAN-t5-base
tokenizer = AutoTokenizer.from_pretrained(model_id)

from datasets import load_dataset
from datasets import Dataset,DatasetDict
dataset = load_dataset("gopalkalpande/bbc-news-summary")

dataset = DatasetDict({
    'train': dataset['train'],
    'test': dataset['train']
})

print(dataset)

encoder="article"
decoder="highlights"

from datasets import concatenate_datasets

# The maximum total input sequence length after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x[encoder], truncation=True), batched=True, remove_columns=[encoder, decoder])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x[decoder], truncation=True), batched=True, remove_columns=[encoder, decoder])
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")

def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample[encoder]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample[decoder], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=[encoder, decoder])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

from transformers import AutoModelForSeq2SeqLM

# huggingface hub model id
model_id="google/flan-t5-base"

# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Hugging Face repository id
repository_id = f"{model_id.split('/')[1]}-{'bbc'}"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=repository_id,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=1,
    # logging & evaluation strategies
    # logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
#     metric_for_best_model="overall_f1",
    # push to hub parameters
    # report_to="tensorboard",
    # push_to_hub=False,
    # hub_strategy="every_save",
    # hub_model_id=repository_id,
    # hub_token=HfFolder.get_token(),
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset["test"]
)
import os
os.environ["WANDB_DISABLED"] = "true"

# Start training
trainer.train()

from transformers import pipeline
from random import randrange        

# load model and tokenizer from huggingface hub with pipeline
summarizer = pipeline("summarization", model="soumagok/flan-t5-base-cnn-dailymail", device=0)

# select a random test sample
sample = dataset['test'][1]
print(f"dialogue: \n{sample['document']}\n---------------")

# summarize dialogue
res = summarizer("""A fire alarm went off at the Holiday Inn in Hope Street at about 04:20 BST on Saturday and guests were asked to leave the hotel. As they gathered outside they saw the two buses, parked side-by-side in the car park, engulfed by flames. One of the tour groups is from Germany, the other from China and Taiwan. It was their first night in Northern Ireland. The driver of one of the buses said many of the passengers had left personal belongings on board and these had been destroyed. Both groups have organised replacement coaches and will begin their tour of the north coast later than they had planned. Police have appealed for information about the attack. Insp David Gibson said: "It appears as though the fire started under one of the buses before spreading to the second. "While the exact cause is still under investigation, it is thought that the fire was started deliberately""")
print(f"flan-t5-base summary:\n{res}")
