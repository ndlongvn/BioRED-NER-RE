import nltk
from transformers import BertTokenizer
import logging
from torch.utils.data import Dataset
import torch.nn as nn
import os
from typing import Optional, List
logger = logging.getLogger(__name__)
import warnings
import pickle
import torch
import functools
import random
import time
import numpy as np
from transformers import Trainer

#
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

#
def open_pickle(path):
    """Open pickle file"""
    if not os.path.isfile(path):
        raise ValueError("File is not existed.")
    with open(path, 'rb') as f:
        return pickle.load(f)


#
class BaseFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids):
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids

#
class BertFeature(BaseFeature):
    '''
    Input of Bert model
    '''
    def __init__(self, token_ids, attention_masks, token_type_ids, labels=None, ids=None):
        super(BertFeature, self).__init__(
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids)
        self.labels = labels
        # self.ids = ids

#   
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    def __init__(self, input_ids: List[int], attention_mask: List[int], token_type_ids: Optional[List[int]] = None, 
                 label_ids: Optional[int] = None   ):
        self.input_ids= input_ids # list of token ids
        self.attention_mask= attention_mask # list of attention mask
        self.token_type_ids= token_type_ids # list of token type ids
        self.label= label_ids# list of label ids

class REDataset(Dataset):
    """
    Dataset for relation extraction
    """
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    def __init__(
            self,
            data_dir: str,    
            tokenizer: BertTokenizer,                  
            labels: List[str],
            
            load_dir: str= '/media/data3/users/longnd/ehr-relation-extraction/biobert_ner/data/BioRED',
            max_seq_length: Optional[int] = 512,
            overwrite_cache=False,
            mode: str = 'Train',
            ):
        # Load data features from cache or dataset file
        cached_features_file =  data_dir

        if os.path.exists(os.path.join(cached_features_file, mode+ '.pkl')) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                # self.features = torch.load(cached_features_file)
                self.features = open_pickle(os.path.join(cached_features_file, mode+ '.pkl'))
        else:
                logger.info(f"Creating features from dataset file at {load_dir}")
                # load feature from pubtator file
                self.features = pubtator_to_re(pubtator_file=load_dir+'{}.PubTator'.format(mode), tokenizer=tokenizer, labels=labels, max_seq_length=max_seq_length)
                logger.info(f"Saving features into cached file {cached_features_file}"+' for '+ mode+ ' file.')
                save_pickle(self.features, cached_features_file, mode+ '.pkl')
        # self.token_ids = [torch.tensor(example.token_ids).long() for example in self.features]
        # self.attention_masks = [torch.tensor(example.attention_masks).float() for example in self.features]
        # self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in self.features]
        # self.labels = [torch.tensor(example.labels).long() for example in self.features]

    def __len__(self):
        return len(self.features)

    # def __getitem__(self, index):
    #     return {
    #         'token_ids': self.token_ids[index],
    #         'attention_masks': self.attention_masks[index],
    #         'token_type_ids': self.token_type_ids[index],
    #         'labels': self.labels[index],
    #     }
    def __getitem__(self, index)-> InputFeatures:
        return self.features[index]
    
def split_text_by_nltk(text):
    """Split sentence using spacy"""

    sentences = nltk.sent_tokenize(text, language='english')
    # print(sentences)
    start_positions= []
    end_positions = []
    for sentence in sentences:
        start_positions.append(text.index(sentence))
        end_positions.append(text.index(sentence) + len(sentence))
        assert sentence == text[text.index(sentence):text.index(sentence) + len(sentence)]
    return sentences


def convert_raw_text(text:str, list_en1: List[str], list_en2: List[str]):
    list_en= list_en1+ list_en2
    list_en= sorted(list_en, key= lambda x: x[0])   
    new_text= ''
    begin= 0
    for i in list_en:
        assert text[i[0]:i[1]] == i[3]
        new_text+= text[begin:i[0]]
        new_text+= '@'+i[2]+'$'+' '+ i[3]+' '+'@/'+i[2]+'$'
        begin= i[1]
    new_text+= text[begin:]
    return new_text

def convert_raw_text2(text:str, list_en1: List[str], list_en2: List[str]):
    list_en= list_en1+ list_en2
    list_en= sorted(list_en, key= lambda x: x[0])   
    new_text= ''
    begin= 0
    for i in list_en:
        assert text[i[0]:i[1]] == i[3]
        new_text+= text[begin:i[0]]
        new_text+= '@'+i[2]+'$' # thay tên thành type
        begin= i[1]
    new_text+= text[begin:]
    return new_text

def get_special_token():
    return {'@GeneOrGeneProduct$':0, 
            '@DiseaseOrPhenotypicFeature$':0,
            '@ChemicalEntity$':0,
            '@SequenceVariant$':0,
            '@/GeneOrGeneProduct$':1,
            '@/DiseaseOrPhenotypicFeature$':1,
            '@/ChemicalEntity$':1,
            '@/SequenceVariant$':1,}
def get_special_token2():
    return {
            '@GeneOrGeneProduct$':0, 
            '@DiseaseOrPhenotypicFeature$':0,
            '@ChemicalEntity$':0,
            '@SequenceVariant$':0,}

def convert_text2_bertfeature(text:str, label:int, tokenizer:BertTokenizer, max_seq_length:int=512):
    
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
    return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids= token_type_ids,label_ids= label)


def get_relation_pair(ent1, ent2):
    if (ent1, ent2) ==('GeneOrGeneProduct', 'GeneOrGeneProduct'):
        return 'G-G'
    elif (ent1, ent2) ==('GeneOrGeneProduct', 'DiseaseOrPhenotypicFeature') or (ent1, ent2) ==('DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct'):
        return 'G-D'
    elif (ent1, ent2) ==('GeneOrGeneProduct', 'ChemicalEntity') or (ent1, ent2) ==('ChemicalEntity', 'GeneOrGeneProduct'):
        return 'G-C'
    elif (ent1, ent2) ==('DiseaseOrPhenotypicFeature', 'SequenceVariant') or (ent1, ent2) ==('SequenceVariant', 'DiseaseOrPhenotypicFeature'):
        return 'D-V'
    elif (ent1, ent2) ==('ChemicalEntity', 'DiseaseOrPhenotypicFeature') or (ent1, ent2) ==('DiseaseOrPhenotypicFeature', 'ChemicalEntity'):
        return 'C-D'
    elif (ent1, ent2) ==('ChemicalEntity', 'ChemicalEntity'):
        return 'C-C'
    elif (ent1, ent2) ==('ChemicalEntity', 'SequenceVariant') or (ent1, ent2) ==('SequenceVariant', 'ChemicalEntity'):
        return 'C-V'
    elif (ent1, ent2) ==('SequenceVariant', 'SequenceVariant'):
        return 'V-V'
    elif (ent1, ent2) ==('GeneOrGeneProduct', 'SequenceVariant') or (ent1, ent2) ==('SequenceVariant', 'GeneOrGeneProduct'):
        return 'G-V'
def pubtator_to_re(pubtator_file: str,
                   tokenizer: BertTokenizer=None,
                    not_use =['OrganismTaxon', 'CellLine'],
                    max_seq_length: int = 512,                    
                    labels= None,
                    verbose: int = 1
                    ):
    
    feature= []
    relation_pair= [['GeneOrGeneProduct', 'GeneOrGeneProduct'], ['GeneOrGeneProduct', 'DiseaseOrPhenotypicFeature'],
                    ['GeneOrGeneProduct', 'ChemicalEntity'], ['DiseaseOrPhenotypicFeature', 'SequenceVariant'],
                    ['ChemicalEntity', 'DiseaseOrPhenotypicFeature'], ['ChemicalEntity', 'ChemicalEntity'],
                    ['ChemicalEntity', 'SequenceVariant'], ['SequenceVariant', 'SequenceVariant'],
                    ['GeneOrGeneProduct', 'SequenceVariant']]
    
    label_map = {label: i for i, label in enumerate(labels)}
    # Parse PubTator annotations
    ex_index= 0
    # from text of entity to code of it in pubtator
    text2code= {}
    with open(pubtator_file, 'r') as f: 
        pubtator_text = f.read()
    annotations = {}
    for line in pubtator_text.strip().split('\n'):
        fields = line.split('\t')
        if len(fields) ==1:
            if fields[0]== '': 
                continue
            pmid, section, text = fields[0].split("|")
            if section == 't':
                annotations[pmid] = {'text': text, 'entities': [] , 'relations': [], 'ent_type_id': set(), 'pair': []}
                text2code[pmid]= {}
            else:
                annotations[pmid]['text']+= " "+ text
                text2code[pmid]= {}
            continue
        pmid = fields[0]
        if pmid not in annotations:
            annotations[pmid] = {'text': fields[2], 'entities': [], 'relations': [], 'ent_type_id': set(), 'pair': []}
            text2code[pmid]= {}
        if len(fields) == 6:
            start, end = int(fields[1]), int(fields[2])
            entity_type = fields[4]
            if fields[4] in not_use:
                continue
            entity_text= fields[3]
            code = fields[5]
            assert annotations[pmid]['text'][start:end] == entity_text
            # list_type= code.split(',')
            annotations[pmid]['entities'].append((start, end, entity_type, entity_text))
            list_code= code.split(',')
            if len(list_code)>1:
                annotations[pmid]['pair'].append(list_code) # 1 vài entity cấu tạo từ nhiều chất
            for code in list_code:
                text2code[pmid][code] = []
                text2code[pmid][code].append((start, end, entity_type, entity_text))
            # add id of entity type
                annotations[pmid]['ent_type_id'].add(code)
        elif len(fields) == 5:
            re_type= fields[1]
            entity_type1= fields[2]
            entity_type2= fields[3]
            # print(text2code[pmid][entity_type1][0][2], text2code[pmid][entity_type2][0][2])
            if text2code[pmid][entity_type1][0][2] in not_use or text2code[pmid][entity_type2][0][2] in not_use:
                continue
            annotations[pmid]['relations'].append((re_type, entity_type1, entity_type2))

    
    relation = {}
    # load all relation for each sentence
    for pmid in annotations.keys():
        relation[pmid] = []
        annotations[pmid]['ent_type_id'] = list(annotations[pmid]['ent_type_id'])
        use_relation = list()
        use_relation_type= list()
        for (re_ty, ent1_, ent2_) in annotations[pmid]['relations']:
            use_relation.append((ent1_, ent2_))
            use_relation_type.append(re_ty)
        use= 0       
        for i in range(len(annotations[pmid]['ent_type_id'])):
            # load all entity_code
            ent1= annotations[pmid]['ent_type_id'][i]
            ent1_ty= text2code[pmid][ent1][0][2]
            for j in range(i, len(annotations[pmid]['ent_type_id'])):                
                ent2= annotations[pmid]['ent_type_id'][j]                   
                ent2_ty= text2code[pmid][ent2][0][2]
                if not([ent1_ty, ent2_ty] in relation_pair or [ent2_ty, ent1_ty] in relation_pair):
                    continue         
                # if ((ent1_ty, ent2_ty) not in src_tgt_pairs) and ((ent2_ty, ent1_ty) not in src_tgt_pairs):
                #     continue
                if (ent1, ent2) in use_relation:
                    relation[pmid].append((annotations[pmid]['text'], ent1, ent2, use_relation_type[use_relation.index((ent1, ent2))]))
                    use+=1
                elif (ent2, ent1) in use_relation:
                    relation[pmid].append((annotations[pmid]['text'], ent2, ent1, use_relation_type[use_relation.index((ent2, ent1))]))
                    use+=1
                else:
                    relation[pmid].append((annotations[pmid]['text'], ent1, ent2, 'No_Relation'))
        assert len(annotations[pmid]['relations']) == use, (pmid, len(annotations[pmid]['relations']), use)

        # with each pair of entity, we will create a sample input 
        for (text, ent1, ent2, rela_ty) in relation[pmid]:
            ent1_ty= list(text2code[pmid][ent1])
            ent2_ty= list(text2code[pmid][ent2])
            # new convert text
            # new_text= convert_raw_text(text, ent1_ty, ent2_ty)
            new_text= convert_raw_text(text, ent1_ty, ent2_ty)

            # we add sentence that has entity for text feature
            if verbose and ex_index < 5:
                print("*** Example ***")
                print("guid: %s" % (pmid))
                print("text: %s" % (text))
                print('new_text: %s' % (new_text))
                print("entity1: %s" % (ent1))
                print("entity2: %s" % (ent2))
                print("relation: %s" % (rela_ty))
            ex_index += 1           
            feature.append(convert_text2_bertfeature(new_text, label_map[rela_ty], tokenizer, max_seq_length))
    return feature    

def get_pair_relation(pubtator_file: str,
                    not_use =['OrganismTaxon', 'CellLine'],
                    ):
    
    # feature= []
    pair= []
    relation_pair= [['GeneOrGeneProduct', 'GeneOrGeneProduct'], ['GeneOrGeneProduct', 'DiseaseOrPhenotypicFeature'],
                    ['GeneOrGeneProduct', 'ChemicalEntity'], ['DiseaseOrPhenotypicFeature', 'SequenceVariant'],
                    ['ChemicalEntity', 'DiseaseOrPhenotypicFeature'], ['ChemicalEntity', 'ChemicalEntity'],
                    ['ChemicalEntity', 'SequenceVariant'], ['SequenceVariant', 'SequenceVariant'],
                    ['GeneOrGeneProduct', 'SequenceVariant']]
    
    # from text of entity to code of it in pubtator
    text2code= {}
    with open(pubtator_file, 'r') as f: 
        pubtator_text = f.read()
    annotations = {}
    for line in pubtator_text.strip().split('\n'):
        fields = line.split('\t')
        if len(fields) ==1:
            if fields[0]== '': 
                continue
            pmid, section, text = fields[0].split("|")
            if section == 't':
                annotations[pmid] = {'text': text, 'entities': [] , 'relations': [], 'ent_type_id': set(), 'pair': []}
                text2code[pmid]= {}
            else:
                annotations[pmid]['text']+= " "+ text
                text2code[pmid]= {}
            continue
        pmid = fields[0]
        if pmid not in annotations:
            annotations[pmid] = {'text': fields[2], 'entities': [], 'relations': [], 'ent_type_id': set(), 'pair': []}
            text2code[pmid]= {}
        if len(fields) == 6:
            start, end = int(fields[1]), int(fields[2])
            entity_type = fields[4]
            if fields[4] in not_use:
                continue
            entity_text= fields[3]
            code = fields[5]
            assert annotations[pmid]['text'][start:end] == entity_text
            # list_type= code.split(',')
            annotations[pmid]['entities'].append((start, end, entity_type, entity_text))
            list_code= code.split(',')
            if len(list_code)>1:
                annotations[pmid]['pair'].append(list_code) # 1 vài entity cấu tạo từ nhiều chất
            for code in list_code:
                text2code[pmid][code] = []
                text2code[pmid][code].append((start, end, entity_type, entity_text))
            # add id of entity type
                annotations[pmid]['ent_type_id'].add(code)
        elif len(fields) == 5:
            re_type= fields[1]
            entity_type1= fields[2]
            entity_type2= fields[3]
            # print(text2code[pmid][entity_type1][0][2], text2code[pmid][entity_type2][0][2])
            if text2code[pmid][entity_type1][0][2] in not_use or text2code[pmid][entity_type2][0][2] in not_use:
                continue
            annotations[pmid]['relations'].append((re_type, entity_type1, entity_type2))

    
    relation = {}
    # load all relation for each sentence
    for pmid in annotations.keys():
        relation[pmid] = []
        annotations[pmid]['ent_type_id'] = list(annotations[pmid]['ent_type_id'])
        use_relation = list()
        use_relation_type= list()
        for (re_ty, ent1_, ent2_) in annotations[pmid]['relations']:
            use_relation.append((ent1_, ent2_))
            use_relation_type.append(re_ty)
        use= 0       
        for i in range(len(annotations[pmid]['ent_type_id'])):
            # load all entity_code
            ent1= annotations[pmid]['ent_type_id'][i]
            ent1_ty= text2code[pmid][ent1][0][2]
            for j in range(i, len(annotations[pmid]['ent_type_id'])):                
                ent2= annotations[pmid]['ent_type_id'][j]                   
                ent2_ty= text2code[pmid][ent2][0][2]
                if not([ent1_ty, ent2_ty] in relation_pair or [ent2_ty, ent1_ty] in relation_pair):
                    continue         
                # if ((ent1_ty, ent2_ty) not in src_tgt_pairs) and ((ent2_ty, ent1_ty) not in src_tgt_pairs):
                #     continue
                if (ent1, ent2) in use_relation:
                    relation[pmid].append((annotations[pmid]['text'], ent1, ent2, use_relation_type[use_relation.index((ent1, ent2))]))
                    use+=1
                elif (ent2, ent1) in use_relation:
                    relation[pmid].append((annotations[pmid]['text'], ent2, ent1, use_relation_type[use_relation.index((ent2, ent1))]))
                    use+=1
                else:
                    relation[pmid].append((annotations[pmid]['text'], ent1, ent2, 'No_Relation'))
        assert len(annotations[pmid]['relations']) == use, (pmid, len(annotations[pmid]['relations']), use)

        # with each pair of entity, we will create a sample input 
        for (text, ent1, ent2, rela_ty) in relation[pmid]:
            ent1_ty= text2code[pmid][ent1][0][2]
            ent2_ty= text2code[pmid][ent2][0][2]
            pair.append(get_relation_pair(ent1_ty, ent2_ty))
    return pair

def get_labels():
    return ['Association',
            'Bind',
            'Comparison',
            'Conversion',
            'Cotreatment',
            'Drug_Interaction',
            'Negative_Correlation',
            'Positive_Correlation',
            'No_Relation']
def timer(func):
    """
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("{}total time{:.4f}Second".format(func.__name__, end - start))
        return res

    return wrapper


def set_seed(seed=123):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_logger(log_path):
    """
    configure log
    :param log_path:s
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not any(handler.__class__ == logging.FileHandler for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(handler.__class__ == logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
           

    
