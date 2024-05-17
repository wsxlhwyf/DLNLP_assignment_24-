# README -- DLNLP_assignment_23-24
Successfully EC to May 17th 23:59
The chosen challenge is "Pronoun Resolution" from Kaggle. \
This challenge is released 4 years ago. \
This challenge is solved by using Pytorch to build a network with the BERT and MLP.\

# Aim
Identify the target of a pronoun within a text passage. \
The pronoun and two candidate names to which the pronoun could refer are provided. \
Need to decide whether the pronoun refers to name A, name B, or neither. \

# Data
The raw data is Gendered Ambiguous Pronouns (GAP) dataset.

# Network---BERT with MLP
Clone the uncased BERT base from the Hugging Face: https://huggingface.co/google-bert/bert-base-uncased/blob/main/README.md . \
Put the uncased BERT base file in the file A. \
Because of the lack of memory, the fine-tuning part is frozon by using "parameter.requires_grad = False".\
Three layers are built by using MLP. \

# Output
Since there is no label for the test datasets. The validation dataset is used as the test one. \
The validation dataset is not part of the training datasets, which means it could be used as the test one. \

# File
The file A contains the main code of the program. \
Running main.py file could call the main code: bert_classify.py in the file A. \
The Datasets file contains the train/validation/test dataset

# Packages
pandas, os, torch, transformers, matplotlib are required.