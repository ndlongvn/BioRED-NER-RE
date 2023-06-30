from ultil_re import *
from typing import Dict
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification, InputFeatures, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from function import *


def main():
    """
    this code using 2 model:
        - model1: predict the pair of entity is relation or not (0: no relation, 1: has relation)
        - model2: predict the type of relation (8 types)
    """
    # Load model and tokenizer

    device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parent_dir = os.path.dirname(os.getcwd())
    model_path1= os.path.join(parent_dir, 'model', 'PubmedBert_Task1') 
    model_path2= os.path.join(parent_dir, 'model', 'PubmedBert_Task2')
    tokenize_path= os.path.join(parent_dir, 'model', 'PubmedBertTokenizer')

    label1= ['No_Relation', 'Relation']
    label2= ['Association', 'Bind', 'Comparison', 'Conversion', 'Cotreatment', 'Drug_Interaction', 'Negative_Correlation', 'Positive_Correlation']
    label_map1: Dict[int, str] = {i: label for i, label in enumerate(label1)}
    label_map2: Dict[int, str] = {i: label for i, label in enumerate(label2)}
    num_label1= len(label1)
    num_label2= len(label2)

    label_map= {i: label for i, label in enumerate(get_labels())}

    config1 = AutoConfig.from_pretrained(model_path1, num_labels=num_label1, 
                                        id2label=label_map1, label2id={label: i for i, label in enumerate(label1)})
    tokenizer = AutoTokenizer.from_pretrained(tokenize_path)

    model1 = AutoModelForSequenceClassification.from_pretrained(model_path1, config=config1)

    config2 = AutoConfig.from_pretrained(model_path2, num_labels=num_label2, 
                                        id2label=label_map2, label2id={label: i for i, label in enumerate(label2)})

    model2 = AutoModelForSequenceClassification.from_pretrained(model_path2, config=config2)

    # Load data
    text= ""

    """
    text: raw text
    """
    entity_list= () # as form: {code: (N, start, end, type, ent_text)}
    # about entity_list:

    """
    code: code of entity
    N: N entity has same code (because one code can have many entity)
    start: start of entity
    end: end of entity
    type: type of entity
    ent_text: text of entity
    """

    # Convert raw text to input features
    feature, code_pair= convert_text_to_input_bert(text= text, entity_list= entity_list, tokenizer= tokenizer, max_seq_length= 512)
    test_dataset= REDatasetForPredict(features= feature)
    data= DataLoader(test_dataset, batch_size= 1, shuffle= False, num_workers= 4)

    # Predict
    pred= predict(model1, model2, data, device)

    # Result
    result= []
    for i in range(len(pred)):
        if pred[i]!= 8:
            result.append((code_pair[i], label_map[pred[i]]))
    """
    result: list of tuple (code_pair, relation)
    """


    






