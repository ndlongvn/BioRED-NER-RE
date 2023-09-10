
# Name Entity Recognition and Relation Extraction for BioRED dataset

  

## Authors

[LongNguyen](https://github.com/ndlongvn)

  

## Content

[NER-RE-BioRED](#ner-re-biored)

-  [Introduction](#introduction)

-  [How to install the environment](#how-to-install-environment)

-  [BioRED dataset ](#biored-dataset)
	- [Download dataset](#download-dataset)
	- [Dataset preprocessing](#dataset-preprocessing)
- [Model](#model)
	- [Download model](#download-model)
	- [Training model](#training-model)
	- [Evaluation](#evaluation)
	- [Testing](#testing)
-  [Web App](#web-app)
- [Citation](#citation)
- [Contact](#contact)

  

### Introduction
Named Entity Recognition (NER) and Relation Extraction (RE) are two essential techniques in the field of biology for extracting structured information from unstructured biological texts. NER focuses on identifying and categorizing specific named entities such as genes, proteins, diseases, and chemicals. It helps in accurately recognizing and classifying these entities, enabling efficient information retrieval and analysis. On the other hand, RE aims to identify relationships or associations between these entities, such as protein-protein interactions or gene-disease relationships. By identifying and classifying these relationships, RE contributes to building knowledge graphs or networks that capture complex biological interactions. The integration of NER and RE allows for the extraction of valuable insights from vast amounts of biological data, facilitating biomedical research, drug discovery, and advancing our understanding of biological systems.

This is the repository for my NLP project at HUST. The project focuses on the task of named entity recognition (NER) and relation extraction (RE) on the BioRED dataset. In this repository, I provide the data processing, training, evaluation, and testing code for new text. Additionally, I have included a simple web app for utilizing the NER system.
### How to install environment

1. Clone the repository into a directory. We refer to that directory as *BIORED-NER-RE-ROOT*.

```Shell
git clone https://github.com/ndlongvn/BioRED-NER-RE.git
```

2. Install all required packages.

```Shell
pip install -r requirements.txt
```
### BioRED dataset
#### Download dataset 
You can download BioRED dataset in [here](https://github.com/ncbi/BioRED).
#### Dataset preprocessing
In NER and RE folder, you can can preprocessing BioRED data to input of BERT model by running:
```
bash rungeneratedata.sh
```
- In this bash code, you should add some parameters as:
	- input_dir: folder to BioRED dataset
	- target_dir: folder to save preprocessing data
	- max_seq_len: max length of input token
	- tokenizer: folder save tokenizer using

### Model
For this project, i using two BERT model: BioBERT and PubMedBERT model, both of them pre-training on biology papers from PubMed. These BERT-based models have been specifically trained on a large corpus of biomedical literature, making them well-suited for tasks in the biological domain.
#### Download model
You can download two model checkpoint by access [Hugging Face – The AI community building the future.](https://huggingface.co/)
#### Training model
For both NER and RE task, you can training model by running:
```
bash runcode.sh

```
With NER, i providing training with focal loss, weight loss and crf layer. You can modify runcode file for using this.
#### Evaluation
For evaluation model after training, you can using Notebook file [eval_ner.ipynb](https://github.com/ndlongvn/BioRED-NER-RE/blob/main/NER%20BioRED/eval_ner.ipynb) and [eval_re.ipynb](https://github.com/ndlongvn/BioRED-NER-RE/blob/main/RE%20BioRED/eval_re.ipynb). 
#### Testing
For testing for new text, you can running code:
```
python test_ner.py
python test_re.py
```
### Web App
This project has simple web app code using for NER task. You can using this app by running main file in NER folder:
```
streamlit run NER_BioRED/main.py
```
### Citation
```bibtex
@article{luo2022biored,
  author    = {Luo, Ling and Lai, Po-Ting and Wei, Chih-Hsuan and Arighi, Cecilia N and Lu, Zhiyong},
  title     = {BioRED: A Rich Biomedical Relation Extraction Dataset},
  journal   = {Briefing in Bioinformatics},
  year      = {2022},
  publisher = {Oxford University Press}
}

@article{lee2020biobert,
  title={BioBERT: a pre-trained biomedical language representation model for biomedical text mining},
  author={Lee, Jinhyuk and Yoon, Wonjin and Kim, Sungdong and Kim, Donghyeon and Kim, Sunkyu and So, Chan Ho and Kang, Jaewoo},
  journal={Bioinformatics},
  volume={36},
  number={4},
  pages={1234--1240},
  year={2020},
  publisher={Oxford University Press}
}

@article{gu2021domain,
  title={Domain-specific language model pretraining for biomedical natural language processing},
  author={Gu, Yu and Tinn, Robert and Cheng, Hao and Lucas, Michael and Usuyama, Naoto and Liu, Xiaodong and Naumann, Tristan and Gao, Jianfeng and Poon, Hoifung},
  journal={ACM Transactions on Computing for Healthcare (HEALTH)},
  volume={3},
  number={1},
  pages={1--23},
  year={2021},
  publisher={ACM New York, NY}
}
```
### Contact
If you have any question, please contact me via [email](long.nd204580@sis.hust.edu.vn).
