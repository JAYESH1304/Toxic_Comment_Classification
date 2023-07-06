# Importing necessary libraries
import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

st.title('Toxic Comment Classification')
comment = st.text_area("Enter Your Text", "Type Here")

comment_input = []
comment_input.append(comment)
test_df = pd.DataFrame()
test_df['comment_text'] = comment_input
cols = {'toxic':[0], 'severe_toxic':[0], 'obscene':[0], 'threat':[0], 'insult':[0], 'identity_hate':[0], 'non_toxic': [0]}
for key in cols.keys():
    test_df[key] = cols[key]
test_df = test_df.reset_index()
test_df.drop(columns=["index"], inplace=True)

# Data Cleaning and Preprocessing
# creating copy of data for data cleaning and preprocessing
cleaned_data = test_df.copy()

# Removing Hyperlinks from text
cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"https?://\S+|www\.\S+","",x) )

# Removing emojis from text
cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub("["
                                                                                   u"\U0001F600-\U0001F64F"
                                                                                   u"\U0001F300-\U0001F5FF"  
                                                                                   u"\U0001F680-\U0001F6FF" 
                                                                                   u"\U0001F1E0-\U0001F1FF"  
                                                                                   u"\U00002702-\U000027B0"
                                                                                   u"\U000024C2-\U0001F251"
                                                                                   "]+","", x, flags=re.UNICODE))

# Removing IP addresses from text 
cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",x))

# Removing html tags from text 
cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"<.*?>","",x))

# There are some comments which contain double quoted words like --> ""words""  we will convert these to --> "words" 
cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"\"\"", "\"",x))   # replacing "" with "
cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"^\"", "",x))      # removing quotation from start and the end of the string
cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"\"$", "",x))

# Removing Punctuation / Special characters (;:'".?@!%&*+) which appears more than twice in the text 
cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"[^a-zA-Z0-9\s][^a-zA-Z0-9\s]+", " ",x))

# Removing Special characters 
cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"[^a-zA-Z0-9\s\"\',:;?!.()]", " ",x))

# Removing extra spaces in text
cleaned_data["comment_text"] = cleaned_data["comment_text"].map(lambda x: re.sub(r"\s\s+", " ",x))

Final_data = cleaned_data.copy()

# Model Building
from transformers import DistilBertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Using Pretrained DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Creating Dataset class for Toxic comments and Labels 
class Toxic_Dataset(Dataset):
    def __init__(self, Comments_, Labels_):
        self.comments = Comments_.copy()
        self.labels = Labels_.copy()
        
        self.comments["comment_text"] = self.comments["comment_text"].map(lambda x: tokenizer(x, padding="max_length", truncation=True, return_tensors="pt"))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        comment = self.comments.loc[idx,"comment_text"]
        label = np.array(self.labels.loc[idx,:])
        
        return comment, label

X_test = pd.DataFrame(test_df.iloc[:, 0])
Y_test = test_df.iloc[:, 1:]
Test_data = Toxic_Dataset(X_test, Y_test)
Test_Loader = DataLoader(Test_data, shuffle=False)

# Loading pre-trained weights of DistilBert model for sequence classification
# and changing classifiers output to 7 because we have 7 labels to classify.
# DistilBERT

from transformers import DistilBertForSequenceClassification

Distil_bert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

Distil_bert.classifier = nn.Sequential(
                    nn.Linear(768,7),
                    nn.Sigmoid()
                  )
# print(Distil_bert)

# Instantiating the model and loading the weights
model = Distil_bert
model.to('cpu')
model = torch.load('dsbert_toxic_balanced.pt', map_location=torch.device('cpu'))

# Making Predictions
for comments, labels in Test_Loader:
    labels = labels.to('cpu')
    labels = labels.float()
    masks = comments['attention_mask'].squeeze(1).to('cpu')
    input_ids = comments['input_ids'].squeeze(1).to('cpu')
    
    output = model(input_ids, masks)
    op = output.logits
    
    res = []
    for i in range(7):
        res.append(op[0, i])
    # print(res)

preds = []

for i in range(len(res)):
    preds.append(res[i].tolist())

classes = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate', 'Non Toxic']

if st.button('Classify'):
    for i in range(len(res)):
        st.write(f"{classes[i]} : {round(preds[i], 2)}\n")
    st.success('These are the outputs')
    
