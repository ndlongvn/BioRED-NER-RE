from ultil_ner import InputFeatures, split_text_by_spacy_example
from transformers import BertTokenizer, Trainer, AutoConfig, AutoTokenizer, AutoModelForTokenClassification
import numpy as np
from torch import nn
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from tqdm import tqdm

"""
mô hình đang bị nhầm lẫn B với I:
    có 2 lựa chọn:
        1. nếu sau I mà có B cùng nhãn thì đổi B thành I
        2. giữ nguyên thì khi detok sẽ tạo ra 2 entity cùng nhãn
"""
class NerDatasetForPredict(Dataset):
    """
    """
    
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    def __init__(self, features: List[InputFeatures]):
        self.features = features
        self.input_ids = [torch.tensor(example.input_ids).long() for example in self.features]
        self.attention_masks = [torch.tensor(example.attention_mask).float() for example in self.features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in self.features]
        self.labels = [torch.tensor(example.label_ids).long() for example in self.features]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_masks": self.attention_masks[i],
            "token_type_ids": self.token_type_ids[i],
            "labels": self.labels[i],
        }
    

def convert_text_to_input_bert(text, tokenizer, labels_map, max_seq_length=512):
    """
    Input: 
        text: raw text
        tokenizer: BertTokenizer
        labels_map: dict{label: index}
        max_seq_length: max length of input
    Output:
        sentences: list of sentences
        sentences_input: list of InputFeatures for Bert Input
    """
    sentences= split_text_by_spacy_example(text)
    sentences_input= []
    for sentence in sentences:
        encoded_input = tokenizer.encode_plus(
                    sentence,
                    add_special_tokens=True,
                    truncation=True,
                    padding="max_length",
                    max_length= max_seq_length,
                    return_attention_mask=True,
                    return_token_type_ids=True)
        input_ids = encoded_input["input_ids"]
        token_type_ids = encoded_input["token_type_ids"]
        attention_mask = encoded_input["attention_mask"]
        label_ids = ['O'] * len(input_ids)
        label_ids= [labels_map[label] for label in label_ids]
        assert len(input_ids) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length

        sentences_input.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, 
                                             token_type_ids=token_type_ids, label_ids=label_ids))
    return sentences, sentences_input

def predict(model, data_loader, label_map, device= 'cpu'):
    '''Model: enity recognition
    label_map: dict{index: label}
    
    Output:
        preds_list: list of list of label of each sentence in B-I-O format
        batch_preds: list of list of index of label of each sentence in digit format
        '''
    model = model.to(device)
    batch_preds = []
    attention_masks = []
    labels = []
    for batch in tqdm(data_loader):
        with torch.no_grad():
            outputs = model(batch['input_ids'].to(device), batch['attention_masks'].to(device), batch['token_type_ids'].to(device))
            preds= np.argmax(outputs[0].detach().cpu().numpy(), axis=2)            
            batch_preds.extend(preds)
            attention_masks.extend(batch['attention_masks'].detach().cpu().numpy())
            labels.extend(batch['labels'].detach().cpu().numpy())
    batch_size, seq_len = np.array(batch_preds).shape
    preds_list = [[] for _ in range(batch_size)]
    labels_list = [[] for _ in range(batch_size)]

    for i in range(batch_size): # số lượng câu chứ không phải batch-size nhé
        for j in range(seq_len):
            if attention_masks[i][j] != 0:
                preds_list[i].append(label_map[batch_preds[i][j]])  
                labels_list[i].append(label_map[labels[i][j]])                 
    return preds_list, batch_preds, labels_list

def predict_example(model, text: str, tokenizer: BertTokenizer, labels, max_seq_length=512, device= 'cpu'):
    """label_map: """
    labels_map = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in labels_map.items()}
    sentences, sentences_input= convert_text_to_input_bert(text, tokenizer, labels_map, max_seq_length)

    test_data= NerDatasetForPredict(sentences_input)
    data= DataLoader(test_data, batch_size= 8, shuffle= False, num_workers= 4)

    preds_list, batch_preds, labels_list= predict(model= model, data_loader= data, label_map= id2label, device= device)
    batch_size, seq_len = np.array(batch_preds).shape
    assert batch_size == len(sentences)
    para= ''
    result= []
    for i in range(batch_size):
        origin_token= tokenizer.tokenize(sentences[i], add_special_tokens=True) # token các từ
        pred_token= preds_list[i] #[0: len(origin_token)] # nhãn của từng token
        new_seq= tokenizer.decode(tokenizer.encode(sentences[i], add_special_tokens=False))
        new_seq= new_seq[0].upper() + new_seq[1:]
        if i==0:
            para+= new_seq
        else:
            para+=' '+ new_seq
        entity_list= []
        entity= ''
        entity_type= ''
        # bắt đầu lấy ra kết quả
        for j in range(len(origin_token)):
                # nếu pred là O thì lưu entity cũ và bỏ qua
                if pred_token[j] == 'O':
                    if entity != '':
                        entity_list.append((entity, entity_type))
                        entity = ''
                        entity_type = ''
                    continue
                else:
                    # bắt đầu bằng B-
                    if pred_token[j].startswith('B-'):
                        # đã có rồi thì thêm vào entity list
                        if entity != '':
                            entity_list.append((entity, entity_type))
                            
                        # gán cho biến mới
                        entity = origin_token[j]
                        if origin_token[j].startswith('##'):
                            entity = origin_token[j][2:]
                        entity_type = pred_token[j][2:]
                        continue

                    # nếu bắt đầu bằng I-
                    elif pred_token[j].startswith('I-'):
                        if entity != '' and entity_type == pred_token[j][2:]:
                            if origin_token[j].startswith('##'):
                                entity+= origin_token[j][2:]
                            else:
                                entity+=  ' '
                                entity+=  origin_token[j]
                            continue

                        elif entity != '' and entity_type != pred_token[j][2:]:
                            entity_list.append((entity, entity_type))
                            entity = origin_token[j]
                            if origin_token[j].startswith('##'):
                                entity = origin_token[j][2:]                            
                            entity_type = pred_token[j][2:]
                            continue

                        elif entity == '' and entity_type == '':
                            entity = origin_token[j]
                            if origin_token[j].startswith('##'):
                                entity = origin_token[j][2:]
                            entity_type = pred_token[j][2:]
                            continue
        result.extend(entity_list)
    return result, para

def predict_2(model, data_loader, label_map, device= 'cpu'):
    '''Model: enity recognition
    label_map: dict{index: label}
    
    Output:
        preds_list: list of list of label of each sentence in B-I-O format
        batch_preds: list of list of index of label of each sentence in digit format'''
    model = model.to(device)
    batch_preds = []
    attention_masks = []
    labels = []
    for batch in tqdm(data_loader):
        with torch.no_grad():
            outputs = model(batch['input_ids'].to(device), batch['attention_masks'].to(device), batch['token_type_ids'].to(device))
            preds= np.argmax(outputs[0].detach().cpu().numpy(), axis=2)            
            batch_preds.extend(preds)
            attention_masks.extend(batch['attention_masks'].detach().cpu().numpy())
            labels.extend(batch['labels'].detach().cpu().numpy())
    batch_size, seq_len = np.array(batch_preds).shape
    preds_list = [[] for _ in range(batch_size)]
    labels_list = [[] for _ in range(batch_size)]

    for i in range(batch_size): # số lượng câu chứ không phải batch-size nhé
        for j in range(seq_len):
            if attention_masks[i][j] != 0:
                preds_list[i].append(label_map[batch_preds[i][j]]) 
                labels_list[i].append(label_map[labels[i][j]])

    previous_label = 'O'
    for i in range(batch_size): # số lượng câu chứ không phải batch-size nhé
        # thử chuyển nếu B ngay sau I thì thành I hoặc I sau O thì thành B xem kết quả thế nào? 
        for j in range(seq_len):
            if attention_masks[i][j] != 0:
                # I ngay sau O thì thành B
                if preds_list[i][j].startswith('I-'):
                    if previous_label == 'O':
                        preds_list[i][j] = 'B-' + preds_list[i][j][2:]
                    previous_label = preds_list[i][j]
                    continue
                if preds_list[i][j].startswith('B-'):
                    if previous_label == 'I-' + preds_list[i][j][2:]:
                        preds_list[i][j] = 'I-' + preds_list[i][j][2:]
                    previous_label = preds_list[i][j]
                    continue
                if preds_list[i][j]=='O':
                    previous_label = 'O'
                    continue
                      
    return preds_list, batch_preds, labels_list

def get_entity(pubtator_file: str, id:int,
                not_use: List[str] =['OrganismTaxon', 'CellLine']):

    with open(pubtator_file, 'r') as f: 
        pubtator_text = f.read()
    annotations = {}
    for line in pubtator_text.strip().split('\n'):
        fields = line.split('\t')
        if len(fields) ==1:
            if fields[0]== '': 
                continue
            pubmed_id, section, text = fields[0].split("|")
            if section == 't':
                annotations[pubmed_id] = {'text': text, 'entities': []}
            else:
                annotations[pubmed_id]['text']+= " "+ text
            continue
        pmid = fields[0]
        if pmid not in annotations:
            annotations[pmid] = {'text': fields[2], 'entities': []}
        if len(fields) == 6:
            start, end = int(fields[1]), int(fields[2])
            entity_type = fields[4]
            if fields[4] in not_use:
                continue
            entity_text= fields[3]
            assert annotations[pmid]['text'][start:end] == entity_text
            annotations[pmid]['entities'].append((entity_text, entity_type))
    return annotations[id]['text'], annotations[id]['entities']
