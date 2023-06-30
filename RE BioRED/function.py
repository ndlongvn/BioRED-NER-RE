import torch
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch import nn
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    def __init__(self, input_ids: List[int], attention_mask: List[int], token_type_ids: Optional[List[int]] = None, 
                 label_ids: Optional[int] = None   ):
        self.input_ids= input_ids # list of token ids
        self.attention_masks= attention_mask # list of attention mask
        self.token_type_ids= token_type_ids # list of token type ids
        self.labels= label_ids # list of label ids
class REDatasetForPredict(Dataset):
    """
    Dataset for relation extraction
    """
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    def __init__(self, features: List[InputFeatures]):
        self.features = features
        self.input_ids = [torch.tensor(example.input_ids).long() for example in self.features]
        self.attention_masks = [torch.tensor(example.attention_mask).float() for example in self.features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in self.features]
        # self.labels = [torch.tensor(example.labels).long() for example in self.features]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index],
            'attention_masks': self.attention_masks[index],
            'token_type_ids': self.token_type_ids[index]
            }

def convert_raw_text(text:str, list_en1: List[str], list_en2: List[str]):
    """
    Input:
        text: raw text 
        list_en1: list of entity1
        list_en2: list of entity2
    Output:
        new_text: add special token for entity
    
    """
    list_en= list_en1+ list_en2
    list_en= sorted(list_en, key= lambda x: x[0])   
    new_text= ''
    begin= 0
    for i in list_en:
        # (start, end, type, ent_text)
        assert text[i[0]:i[1]] == i[3], 'start and end of entity is not correct'
        new_text+= text[begin:i[0]]
        new_text+= '@'+i[2]+'$'+' '+ i[3]+' '+'@/'+i[2]+'$'
        begin= i[1]
    new_text+= text[begin:]
    return new_text

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
            entity1= entity_list[i]
            ent1= entity1[0][2]
            for j in range(i, len(keys)):
                entity2= entity_list[j]
                ent2= entity2[0][2]
                code_list.append(keys[i], keys[j])
                if ((ent1, ent2) not in relation_pair) and ((ent1, ent2) not in relation_pair):
                    continue
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

def predict(model1, model2, data_loader, device= 'cpu'):
    '''Model 1: no relation and relation'''
    model1 = model1.to(device)
    model2 = model2.to(device)
    batch_preds = np.empty((0,), dtype=np.int64)
    
    for batch in tqdm(data_loader):
        with torch.no_grad():
            outputs = model1(batch['input_ids'].to(device), batch['attention_masks'].to(device), batch['token_type_ids'].to(device))
            preds = np.argmax(outputs[0].detach().cpu().numpy(), axis=1)
            if preds[0] == 0:
                preds[0]= 8
            else:
                outputs = model2(batch['input_ids'].to(device), batch['attention_masks'].to(device), batch['token_type_ids'].to(device))
                preds = np.argmax(outputs[0].detach().cpu().numpy(), axis=1)
            batch_preds = np.concatenate((batch_preds, preds), axis=0)           
    return batch_preds

