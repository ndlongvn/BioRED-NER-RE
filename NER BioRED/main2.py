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
Using for display in streamlit
"""

# Load model
with st.spinner('Loading model...'):

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
    time.sleep(0.5)




color = {'GeneOrGeneProduct':'red', 
         'DiseaseOrPhenotypicFeature':'green', 'ChemicalEntity':'blue', 'SequenceVariant':'yellow'}
def highlight_entities(paragraph, result):
    table= []
    highlighted_text = paragraph
    for entities in result.values():
        for entity in entities:
            entity_text = entity[0]
            entity_type = entity[1]
            start = paragraph.find(entity_text)
            end = start + len(entity_text)
            table.append([entity_text, entity_type, start, end])
            highlighted_text = highlighted_text[:start] + f"<span style='color: {color[entity_type]};'>{highlighted_text[start:end]}</span>" + highlighted_text[end:]
    return highlighted_text, table

def main():
    st.title("Named Entity Recognition")

    text = st.text_area("Enter a paragraph:")
    
    if st.button("Analyze"):
        # Apply NER model to identify entities
        with st.spinner('Analyzing...'):
            result, para= predict_example(model= model, text= text, tokenizer= tokenizer, labels= labels, max_seq_length= 512, device= device)

            # Highlight entities in the paragraph
            highlighted_paragraph, table = highlight_entities(para, result)

            # Display the colored paragraph
            st.markdown(highlighted_paragraph, unsafe_allow_html=True)

            # Display the entity details in a table
            entity_table = pd.DataFrame(table, columns=['Text', 'Entity Type', 'Start', 'End']).sort_values(by=['Start'])

            time.sleep(0.5)
        st.table(entity_table)

if __name__ == "__main__":
    main()
