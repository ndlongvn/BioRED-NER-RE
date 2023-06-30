import argparse
import os
from transformers import BertTokenizer
from ultil_ner import get_labels, InputFeatures, pubtator_to_bio, save_pickle, open_pickle
import warnings
warnings.filterwarnings("ignore")
labels= ['B-GeneOrGeneProduct', 'I-GeneOrGeneProduct', 'B-DiseaseOrPhenotypicFeature', \
         'I-DiseaseOrPhenotypicFeature', 'B-ChemicalEntity', 'B-SequenceVariant', 'I-ChemicalEntity', 'I-SequenceVariant', \
           'O']
'''4 classes: disease, gene, chemical, sequence variant'''

    
def parse_arguments():
    """Parses program arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", type=str,
                        help="Directory or name of model Bio-Bert for tokenization. Default is 'dmis-lab/biobert-v1.1'.", # i use biored file
                        default="dmis-lab/biobert-v1.1")
    
    parser.add_argument("--input_dir", type=str,
                        help="Directory with BioRED files. Default is 'data/'.", # i use biored file
                        default="data/")

    parser.add_argument("--target_dir", type=str,
                        help="Directory to save files. Default is 'dataset/'.",
                        default='dataset/')

    parser.add_argument("--max_seq_len", type=int,
                        help="Maximum sequence length. Default is 512.",
                        default=256)
    
    arguments = parser.parse_args()
    return arguments


def generate_input_files(input_dir: str,
                         target_dir: str,
                         max_seq_len: int,
                         tokenizer_link: str,
                         mode: str = 'train',):
    
    """Generates input files for NER task"""
    print("Generating {} files for NER task".format(mode))
    tokenizer= BertTokenizer.from_pretrained(tokenizer_link)
    labels= get_labels()
    data= [InputFeatures(input_ids= [], attention_mask= [], token_type_ids= [], label_ids= [])]
    data= pubtator_to_bio(pubtator_file= input_dir, tokenizer=tokenizer, max_seq_length=max_seq_len, labels=labels)
    save_pickle(data, target_dir, mode+ '.pkl')


def main():
    args= parse_arguments()
    for mode in ['Dev.PubTator', 'Train.PubTator', 'Test.PubTator']:
        generate_input_files(input_dir= os.path.join(args.input_dir, mode),
                             target_dir= args.target_dir,
                             max_seq_len= args.max_seq_len,
                             tokenizer_link= args.tokenizer,
                             mode= mode.split('.')[0])

if __name__ == '__main__':
    main()