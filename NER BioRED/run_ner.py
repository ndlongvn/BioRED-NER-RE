import logging
import os
import sys

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
    IntervalStrategy
)

from ultil_ner import InputFeatures, NerDataset, get_labels

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models",
                  "default": "dmis-lab/biobert-v1.1"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # hyperparameter_tuning: bool = field(default=False, metadata={"help": "Set this flag to use hyperparameter tuning."})

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: str = field(
        metadata={"help": "The input data dir. Contain .plk file of feature."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not load labels default"},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

def optuna_hp_space(trial):
    return {
        "num_train_epochs" : trial.suggest_int("num_train_epochs", 1, 5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 10, 12]),
        "per_device_eval_batch_size": trial.suggest_categorical("per_device_eval_batch_size", [8, 10, 12]),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 1000),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "fp16_opt_level": trial.suggest_categorical("fp16_opt_level", ["O0", "O1", "O2", "O3"]),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.0, 1.0),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-8, 1e-6),
    }
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.load_best_model_at_end=True
    training_args.evaluation_strategy = IntervalStrategy.STEPS, # "steps"
    training_args.eval_steps = 50,

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
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

    # Prepare CONLL-2003 task
    labels = get_labels()
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        # cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        # cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode= 'Train'
            # mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode='Dev',
        )
        if training_args.do_eval
        else None
    )

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) \
            -> Tuple[List[List[str]], List[List[str]]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])
        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        
        return {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }
    def compute_objective(metrics):
        return metrics["eval_f1"]
    early_stopping = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.01)
    # Initialize our Trainer
    # Tuning
    # def model_init():
    #     return AutoModelForTokenClassification.from_pretrained(
    #                                                 model_args.model_name_or_path,
    #                                                 from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #                                                 config=config,
    #                                                 # cache_dir=model_args.cache_dir,
    #                                             )
    # if model_args.hyperparameter_tuning:
    #     trainer = Trainer(
    #         model=None,
    #         args=training_args,
    #         train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
    #         compute_metrics=compute_metrics,
    #         tokenizer=tokenizer,
    #         model_init=model_init,
    #     )
    #     best_trial = trainer.hyperparameter_search(
    #                     direction="maximize",
    #                     backend="optuna",
    #                     hp_space=optuna_hp_space,
    #                     n_trials=20,
    #                     compute_objective=compute_objective,
    #                 )
    #     trainer= best_trial.best_model_trainer
    # else:
    trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                callbacks=[early_stopping],
            )
    if training_args.do_train:
            trainer.train(
                model_path=model_args.model_name_or_path #if os.path.isdir(model_args.model_name_or_path) else None
            )
            trainer.save_model(training_args.output_dir)
            # if trainer.is_world_master():
            #     tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()
        print(result)
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")

        with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        results.update(result)

    # Predict
    if training_args.do_predict:
        test_dataset = NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode='Test'
        )

        predictions, label_ids, metrics = trainer.predict(test_dataset)
        logger.info("Predictions shape: " + str(predictions.shape))

        preds_list, _ = align_predictions(predictions, label_ids)
        
        # Save predictions
        output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        # if trainer.is_world_master():
        with open(output_test_results_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        prev_pred = ""
        with open(output_test_predictions_file, "w") as writer:
                with open(os.path.join(data_args.data_dir, "test.txt"), "r") as f:
                    example_id = 0
                    for line in f:
                        if line.startswith("##"):
                            if prev_pred != "O":
                                prev_pred = "I-" + prev_pred.split('-')[-1]
                            output_line = line.split()[0] + " " + prev_pred + "\n"
                            writer.write(output_line)
                        elif line.startswith("-DOCSTART-") or line == "" or line == "\n":
                            writer.write(line)
                            if not preds_list[example_id]:
                                example_id += 1
                        elif preds_list[example_id]:
                            prev_pred = preds_list[example_id].pop(0)
                            output_line = line.split()[0] + " " + prev_pred + "\n"
                            writer.write(output_line)
                        else:
                            logger.warning(
                                "Example %d, Example: %s" % (example_id, line))
            
    return results


if __name__ == "__main__":
    main()
