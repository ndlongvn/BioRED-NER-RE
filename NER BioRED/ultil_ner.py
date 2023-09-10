import sys
sys.path.append("../")

import logging
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Union, Dict

from filelock import FileLock

from transformers import BertTokenizer
from torch import nn
from torch.utils.data.dataset import Dataset
import pickle

logger = logging.getLogger(__name__)
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")
import spacy
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.utils import logging

# import nltk

def split_text_by_spacy(text, entity= None, max_length=512, single_sentence=True):
    """Split sentence using spacy"""
    max_length = max_length - 2
    nlp = spacy.load("en_core_sci_md")

    sentences = nlp(text).sents
    sentences =[str(i) for i in sentences if len(i) > 1]
    # print(sentences)
    start_positions= []
    end_positions = []
    for sentence in sentences:
        start_positions.append(text.index(sentence))
        end_positions.append(text.index(sentence) + len(sentence))
        assert sentence == text[text.index(sentence):text.index(sentence) + len(sentence)]

    data= {}
    for i in range(len(sentences)):
        data[i] = {'text': sentences[i], 'start': start_positions[i], 'end': end_positions[i], 'entity': []}
        entity_in_sentence = []
        for j in range(len(entity)):
            start, end, entity_type, entity_text= entity[j]
            assert text[start:end] == entity_text
            if start_positions[i] <= start and end <= end_positions[i]:
                entity_in_sentence.append((start- data[i]['start'], end- data[i]['start'], entity_type, entity_text))
        data[i]['entity'] = entity_in_sentence

    if single_sentence:
        return data
    else:
        return 0
def split_text_by_spacy_example(text):
    nlp = spacy.load("en_core_sci_md")

    sentences = nlp(text).sents
    sentences =[str(i) for i in sentences if len(i) > 1]
    return sentences
class Raw_data:
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels


def tokenize_with_positions_by_space(text):
    '''tokenize text by space and return tokens and positions'''
    tokens = []
    positions = []
    for match in re.finditer(r'\S+', text):
        token = match.group(0)
        start = match.start()
        end = match.end()
        tokens.append(token)
        positions.append((start, end))
    return tokens, positions

# label not use
not_use =['OrganismTaxon', 'CellLine']

# convert text to input bert
def convert_text_to_id(text, tokenizer, entities, label_map, max_seq_length=256):
    # Tokenize the text
    """label_map = {label: i for i, label in enumerate(labels)}"""
    tokenized_text = tokenizer.encode(text, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_text)
    encoded_input = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                max_length= max_seq_length,
                return_attention_mask=True,
                return_token_type_ids=True)
    input_ids = encoded_input["input_ids"]
    token_type_ids = encoded_input["token_type_ids"]
    attention_mask = encoded_input["attention_mask"]

    # Create labels for each token
    labels = ['O'] * len(input_ids)
    for start, end, entity_type, entity_text in entities:
        start_token = len(tokenizer.encode(text[:start], add_special_tokens=True))-1
        end_token = len(tokenizer.encode(text[:end], add_special_tokens=True))- 1
        labels[start_token] = 'B-' + entity_type
        for i in range(start_token + 1, end_token):
            labels[i] = 'I-' + entity_type

    # Convert labels to label ids
    label_ids = [label_map[label] for label in labels]
    

    assert len(input_ids) == max_seq_length # input id
    assert len(attention_mask) == max_seq_length # attention mask
    assert len(token_type_ids) == max_seq_length # token type id
    assert len(label_ids) == max_seq_length # label id

    return input_ids, attention_mask, token_type_ids, label_ids, tokenized_text
    

# convert pubtator to bio format
def pubtator_to_bio(pubtator_file: str, tokenizer: BertTokenizer,
                    not_use: List[str] = [], # ['OrganismTaxon', 'CellLine'],
                    max_seq_length: int = 512,                    
                    labels: List[str] = None,
                    verbose: int = 1
                    ):
    feature= []
    label_map = {label: i for i, label in enumerate(labels)}
    # Parse PubTator annotations
    ex_index= 0

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
            annotations[pmid]['entities'].append((start, end, entity_type, entity_text))

    # Split text into sequence and tokenization
    for pmid in annotations.keys():
        # plus ex_index
        ex_index += 1
        # load feature
        text_input = annotations[pmid]['text']
        text_list = split_text_by_spacy(text_input, annotations[pmid]['entities'])
        # if ex_index< 2:
        #     print(text_list)
        for i in text_list.keys():
            text= text_list[i]['text']
            entity= text_list[i]['entity']
            input_ids, attention_mask, token_type_ids, encoded_labels, tokens\
                                        = convert_text_to_id(text, tokenizer, entity, label_map, max_seq_length)

            assert len(input_ids) == max_seq_length
            assert len(attention_mask) == max_seq_length
            assert len(token_type_ids) == max_seq_length
            assert len(encoded_labels) == max_seq_length

            if ex_index < 2 and verbose == 1:
                print("*** Example ***")
                print("guid: %s", pmid+'%{}'.format(i))
                print("tokens: %s", tokens)
                print("input_ids: %s", input_ids)
                print("input_mask: %s", attention_mask)
                print("token_type_ids: %s", token_type_ids)
                print("label_ids: %s", encoded_labels)

            feature.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label_ids=encoded_labels
                )
            )
    return feature


class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    def __init__(self, input_ids: List[int], attention_mask: List[int], token_type_ids: Optional[List[int]] = None, 
                 label_ids: Optional[List[int]] = None   ):
        self.input_ids= input_ids # list of token ids
        self.attention_mask= attention_mask # list of attention mask
        self.token_type_ids= token_type_ids # list of token type ids
        self.label_ids= label_ids# list of label ids


#for input data of bert model
class InputData(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    def __init__(self, features: List[InputFeatures]):
        self.features= features

    def __len__(self):
            return len(self.features)
    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

class NerDataset(Dataset):

    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
            self,
            data_dir: str,           
            tokenizer: BertTokenizer,
            labels: List[str],
            model_type: str,
            load_dir: str= '/media/data3/users/longnd/ehr-relation-extraction/biobert_ner/data/BioRED',
            max_seq_length: Optional[int] = 256,
            overwrite_cache=False,
            mode: str = 'Train',
            ):
        # Load data features from cache or dataset file
        cached_features_file =  data_dir

        if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                # self.features = torch.load(cached_features_file)
                self.features = open_pickle(os.path.join(cached_features_file, mode+ '.pkl'))
        else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                # load feature from pubtator file
                self.features = pubtator_to_bio(pubtator_file= load_dir, tokenizer= tokenizer, labels= labels, max_seq_length= max_seq_length, verbose= 1)
                logger.info(f"Saving features into cached file {cached_features_file}")
                save_pickle(self.features, cached_features_file, mode+ '.pkl')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

# get class labels
def get_labels(path: str= None) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ['B-GeneOrGeneProduct', 'I-GeneOrGeneProduct', 'B-DiseaseOrPhenotypicFeature', 
         'I-DiseaseOrPhenotypicFeature', 'B-ChemicalEntity', 'I-ChemicalEntity', 'B-SequenceVariant', 'I-SequenceVariant',
          'B-OrganismTaxon', 'I-OrganismTaxon', 'B-CellLine', 'I-CellLine',
           'O'] # 'OrganismTaxon', 'CellLine'

def save_pickle(file, path, file_name):
    """Save file to pickle format"""
    if os.path.isfile(os.path.join(path, file_name)):
        warnings.warn("File is existed. Overwriting the file.")
    else:
        if not os.path.isdir(path):
            os.mkdir(path)
    with open(os.path.join(path, file_name), 'wb') as f:
        pickle.dump(file, f)
        print("File is saved in " + file_name)


def open_pickle(path):
    """Open pickle file"""
    if not os.path.isfile(path):
        raise ValueError("File is not existed.")
    with open(path, 'rb') as f:
        return pickle.load(f)



class BioBERTCRF(BertPreTrainedModel):
    def __init__(self, config, model_dir, num_labels=9):
        super().__init__(config)
        self.bert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = nn.CRF(config.num_labels, batch_first=True)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.byte())
            return TokenClassifierOutput(loss=loss, logits=logits)

        return TokenClassifierOutput(logits=logits)
