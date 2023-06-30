from ultil_re import get_labels, REDataset, get_special_token
from typing import Dict, List, Tuple
from transformers import BertTokenizer, Trainer, AutoConfig, AutoTokenizer, AutoModelForTokenClassification
import numpy as np
from torch import nn
from sklearn.metrics import classification_report
model_path= '/media/data3/users/longnd/ehr-relation-extraction/biobert_ner/model/BiomedNLP-PubMedBERT-base-uncased-abstract'
labels = get_labels()
label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
num_labels = len(labels)
# config
config = AutoConfig.from_pretrained( model_path, num_labels=num_labels, 
                                    id2label=label_map, label2id={label: i for i, label in enumerate(labels)})
# tokenizer
additional_special_tokens = [i for i in get_special_token().keys()]
tokenizer= BertTokenizer.from_pretrained(model_path)
tokenizer.add_tokens(additional_special_tokens)
tokenizer.additional_special_tokens = additional_special_tokens
# model
model = AutoModelForTokenClassification.from_pretrained(model_path, config=config)
trainer = Trainer(model=model)
test_dataset= REDataset('/media/data3/users/longnd/ehr-relation-extraction/biobert_re/data/Preprocess_PubmedBert', 
                        tokenizer, labels, mode= 'Test')
# predict
predictions, label_ids, metrics = trainer.predict(test_dataset)
pred = np.argmax(predictions, axis=1)
print(classification_report(y_true=label_ids,y_pred= pred, target_names=labels))
