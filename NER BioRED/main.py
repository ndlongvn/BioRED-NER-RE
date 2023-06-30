import torch
import streamlit as st
from ultil_ner import get_labels
from transformers import  AutoConfig, AutoTokenizer, AutoModelForTokenClassification
import numpy as np
from function import predict_example, NerDatasetForPredict
from typing import Dict
import os
import pandas as pd
import time

"""
Using for predict
"""

def main():
    # Load model
    parent_dir = os.path.dirname(os.getcwd())
    model_path= os.path.join(parent_dir, 'model', 'Biobert') # Biobert đang tốt hơn pubmed bert
    labels = get_labels()
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    config = AutoConfig.from_pretrained( model_path, num_labels=num_labels, 
    id2label=label_map, label2id={label: i for i, label in enumerate(labels)})
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=None)
    model = AutoModelForTokenClassification.from_pretrained(model_path, config=config)
    device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # add data
    text = ""

    # predict
    result, para= predict_example(model= model, text= text, tokenizer= tokenizer, labels= labels, max_seq_length= 512, device= device)
    """
    result: list of entity was predicted
    para: new paragraph from original paragraph, different with original paragraph is that it was detokenized
    ex:
        original paragraph: (STN5)
        new paragraph: ( STN 5 )
    """

