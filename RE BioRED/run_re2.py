
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from transformers import GlueDataTrainingArguments as DataTrainingArguments

from metrics import compute_metrics, compute_metrics_f1
from ultil_re import get_labels, get_special_token, REDataset, save_pickle, BertFeature
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from torch import nn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
    IntervalStrategy
)
from transformers import Trainer
from sklearn.utils import class_weight
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import f1_score




logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models",
                  "default": "dmis-lab/biobert-v1.1"})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # hyperparameter_tuning: bool = field(default=False, metadata={"help": "Set this flag to use hyperparameter tuning."})

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: str = field(
        metadata={"help": "The input data dir. Contain .plk file of feature."})
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not load labels default"},)
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."},)
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    training_args.load_best_model_at_end=True
    training_args.evaluation_strategy = IntervalStrategy.STEPS, # "steps"
    training_args.eval_steps = 50
    training_args.metric_for_best_model= "eval_f1_micro"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    logger.info("Training/evaluation parameters %s", training_args)
    # Set seed
    set_seed(training_args.seed)


    labels = ['No_Relation', 'Relation'] # task1
    # labels = ['Association',
    #         'Bind',
    #         'Comparison',
    #         'Conversion',
    #         'Cotreatment',
    #         'Drug_Interaction',
    #         'Negative_Correlation',
    #         'Positive_Correlation'] # task2
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        use_fast=model_args.use_fast,
    )

    # Get datasets
    # Test oversampling data training

    train_dataset= REDataset(data_args.data_dir, tokenizer, labels, mode= 'Train')
   
    eval_dataset= REDataset(data_args.data_dir, tokenizer, labels, mode= 'Dev')
    # Create a RandomOverSampler instance
    # ros = RandomOverSampler()
    ros = RandomUnderSampler()
    index= np.array([i for i in range(len(train_dataset.features))])
    label= np.array([i.label for i in train_dataset.features])

    # Resample the training data
    X_train_resampled, y_train_resampled = ros.fit_resample(index.reshape(-1, 1), label.reshape(-1, 1))
    # train_dataset.features= [train_dataset.features[i] for i in X_train_resampled.reshape(-1)]
    # compute class weight
    df= pd.DataFrame([])
    df['class']= [i.label for i in train_dataset.features]
    # class_weights
    class_weights=class_weight.compute_class_weight(class_weight='balanced', classes= np.unique(df['class']), y= df['class'])
    # class_weights= np.array([1.0, 7.0])
    print("weight: ", class_weights)
    class_weights=torch.tensor(class_weights, dtype=torch.float).cuda()

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get('logits')
            # compute custom loss
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # logger
    logger.info("Training/evaluation parameters %s", training_args)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,)
    model.resize_token_embeddings(len(tokenizer))

    #compute metrics
    def compute_metrics_fn(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return compute_metrics_f1(preds, p.label_ids)
    early_stopping = EarlyStoppingCallback( early_stopping_patience=5, early_stopping_threshold=0.01)
    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn,
        callbacks=[early_stopping],
        optimizers= (torch.optim.Adam(model.parameters(), lr=5e-5), None)
    )

    # Training
    if training_args.do_train:
        trainer.train()
        # trainer.save_model(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        result = trainer.evaluate()
        print(result)
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")

        with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        results.update(result)

    # if training_args.do_predict:
    #     logging.info("*** Test ***")
    #     test_datasets = [test_dataset]

    #     for test_dataset in test_datasets:
    #         predictions = trainer.predict(test_dataset=test_dataset).predictions
    #         if output_mode == "classification":
    #             predictions = np.argmax(predictions, axis=1)

    #         output_test_file = os.path.join(
    #             training_args.output_dir,
    #             f"test_predictions.txt"
    #             )
    #         if trainer.is_world_master():
    #             with open(output_test_file, "w") as writer:
    #                 logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
    #                 writer.write("index\tprediction\n")
    #                 for index, item in enumerate(predictions):
    #                     item = test_dataset.get_labels()[item]
    #                     writer.write("%d\t%s\n" % (index, item))

    #             output_label_file = os.path.join(
    #                 data_args.data_dir,
    #                 f"test_labels.tsv"
    #                 )

    #             output_test_result_file = os.path.join(
    #                 training_args.output_dir,
    #                 f"test_results.txt"
    #                 )

    #             test_result = get_eval_results(output_label_file, output_test_file)
    #             with open(output_test_result_file, "w") as writer:
    #                 for key, value in test_result.items():
    #                     logger.info("  %s = %s", key, value)
    #                     writer.write("%s = %s\n" % (key, value))

    return results


if __name__ == "__main__":
    main()
