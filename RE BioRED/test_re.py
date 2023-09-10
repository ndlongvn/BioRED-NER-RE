# from ultil_re import get_labels, REDataset, get_special_token
# from typing import Dict, List, Tuple
# from transformers import BertTokenizer, Trainer, AutoConfig, AutoTokenizer, AutoModelForTokenClassification
# import numpy as np
# from torch import nn
# from sklearn.metrics import classification_report
# model_path= '/media/data3/users/longnd/ehr-relation-extraction/biobert_ner/model/BiomedNLP-PubMedBERT-base-uncased-abstract'
# labels = get_labels()
# label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
# num_labels = len(labels)
# # config
# config = AutoConfig.from_pretrained( model_path, num_labels=num_labels, 
#                                     id2label=label_map, label2id={label: i for i, label in enumerate(labels)})
# # tokenizer
# additional_special_tokens = [i for i in get_special_token().keys()]
# tokenizer= BertTokenizer.from_pretrained(model_path)
# tokenizer.add_tokens(additional_special_tokens)
# tokenizer.additional_special_tokens = additional_special_tokens
# # model
# model = AutoModelForTokenClassification.from_pretrained(model_path, config=config)
# trainer = Trainer(model=model)
# test_dataset= REDataset('/media/data3/users/longnd/ehr-relation-extraction/biobert_re/data/Preprocess_PubmedBert', 
#                         tokenizer, labels, mode= 'Test')
# # predict
# predictions, label_ids, metrics = trainer.predict(test_dataset)
# pred = np.argmax(predictions, axis=1)
# print(classification_report(y_true=label_ids,y_pred= pred, target_names=labels))

from ultil_re import *
from typing import Dict
from transformers import AutoConfig, AutoTokenizer,  AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from function import *

device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device= 'cpu'
parent_dir = os.path.dirname(os.getcwd())
model_path1= os.path.join(parent_dir, 'RE BioRED/model', 'PubmedBert_Task1') 
model_path2= os.path.join(parent_dir, 'RE BioRED/model', 'PubmedBert_Task2')
tokenize_path= os.path.join(parent_dir, 'RE BioRED/model', 'PubmedBertTokenizer')

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
text= text= "Hepatocyte nuclear factor - 6 : associations between genetic variability and type II diabetes and between genetic variability and estimates of insulin secretion."

"""
text: raw text
"""
entity_list= {'3175':[(0,29,'GeneOrGeneProduct','Hepatocyte nuclear factor - 6')],
  'D003924':[(77,93,'DiseaseOrPhenotypicFeature','type II diabetes')],
  '3630':[(143,150,'GeneOrGeneProduct','insulin')]} # as form: {code: (N, start, end, type, ent_text)}
# about entity_list:

"""
code: code of entity
N: N entity has same code (because one code can have many entity)
start: start of entity
end: end of entity
type: type of entity
ent_text: text of entity
"""

def convert_text_to_input_bert(text, entity_list, tokenizer, max_seq_length=512):
    '''
    Input:
        text: raw text
        entity_list: {code: (N, start, end, type, ent_text)}
        tokenizer: tokenizer
        max_seq_length: max length of sequence
    Output:
        feature: list of InputFeatures
        code_list: list of code pair
    '''
    relation_pair= [['GeneOrGeneProduct', 'GeneOrGeneProduct'], ['GeneOrGeneProduct', 'DiseaseOrPhenotypicFeature'],
                    ['GeneOrGeneProduct', 'ChemicalEntity'], ['DiseaseOrPhenotypicFeature', 'SequenceVariant'],
                    ['ChemicalEntity', 'DiseaseOrPhenotypicFeature'], ['ChemicalEntity', 'ChemicalEntity'],
                    ['ChemicalEntity', 'SequenceVariant'], ['SequenceVariant', 'SequenceVariant'],
                    ['GeneOrGeneProduct', 'SequenceVariant']]
    feature= []
    # entity_list: {code: (N, start, end, type, ent_text)}
    code_list = []
    keys= list(entity_list.keys())
    for i in range(len(keys)):
        entity1= entity_list[keys[i]]
        ent1= entity1[0][2]
        for j in range(i, len(keys)):
            entity2= entity_list[keys[j]]
            ent2= entity2[0][2]
            if ([ent1, ent2] not in relation_pair) and ([ent2, ent1] not in relation_pair):
                continue
            code_list.append((keys[i], keys[j]))
            sequence= convert_raw_text(text, entity1, entity2)
            encoded_input = tokenizer.encode_plus(
                                sequence,
                                add_special_tokens=True,
                                truncation=True,
                                padding="max_length",
                                max_length= max_seq_length,
                                return_attention_mask=True,
                                return_token_type_ids=True)
            input_ids = encoded_input["input_ids"]
            token_type_ids = encoded_input["token_type_ids"]
            attention_mask = encoded_input["attention_mask"]
            feature.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids= token_type_ids, label_ids= 0)) 
    return feature, code_list

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
print(result)