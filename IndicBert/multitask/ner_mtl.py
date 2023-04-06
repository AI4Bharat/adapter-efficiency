from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch import nn
import copy, transformers
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer, EvalPrediction
from datasets import load_metric
import numpy as np
from transformers import EarlyStoppingCallback, IntervalStrategy
import os
import argparse
import wandb
#The labels for the NER task and the dictionaries to map the to ids or 
#the other way around
#os.environ["WANDB_MODE"] = "offline"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--model_path", type=str, default="callbacks_MTL_apart__ner_32_3e-05_7")
parser.add_argument("--checkpont_p", type=str, default="checkpoint-69861")
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--max_seq_length", type=int, default=256)
args = parser.parse_args()

wandb.init(project="indicbert_mlm_ner", entity="nandinimundra", name = f"last_please_{args.model_path}_{args.checkpont_p}")

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models. 

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        print("what cls is \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ ", cls)
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name, 
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)

model_name = 'ai4bharat/IndicBERT-MLM-only'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
labels_ner = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
id_2_label_ner = {id_: label for id_, label in enumerate(labels_ner)}
label_2_id_ner = {label: id_ for id_, label in enumerate(labels_ner)}

multitask_model = MultitaskModel.create(
    model_name=model_name,
    model_type_dict={
        "xnli": transformers.AutoModelForSequenceClassification,
        "ner": transformers.AutoModelForTokenClassification,
        "paraphrase": transformers.AutoModelForSequenceClassification,
        "sentiment":transformers.AutoModelForSequenceClassification,
        "qa" : transformers.AutoModelForQuestionAnswering,
        "copa" : transformers.AutoModelForMultipleChoice,
    },
    model_config_dict={
        "xnli": transformers.AutoConfig.from_pretrained(model_name, num_labels=3, use_auth_token=True),
        "ner": transformers.AutoConfig.from_pretrained(model_name, num_labels=7, label2id=label_2_id_ner, id2label=id_2_label_ner, use_auth_token=True),
        "paraphrase": transformers.AutoConfig.from_pretrained(model_name, num_labels=2, use_auth_token=True),
        "sentiment": transformers.AutoConfig.from_pretrained(model_name, num_labels=2, use_auth_token=True), 
        "qa": transformers.AutoConfig.from_pretrained(model_name, use_auth_token= True ),
        "copa" : transformers.AutoConfig.from_pretrained(model_name, use_auth_token= True ),
    },
)
#multitask_model = torch.load(args.model_path)
multitask_model.load_state_dict(torch.load(f"/nlsasfs/home/ai4bharat/nandinim/nandini/new_finetune/MTL/{args.model_path}/{args.checkpont_p}/pytorch_model.bin" ))


labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
id_2_label = {id_: label for id_, label in enumerate(labels)}
label_2_id = {label: id_ for id_, label in enumerate(labels)}

model_name = 'ai4bharat/IndicBERT-MLM-TLM'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)


dataset = load_dataset("conll2003")
dataset = dataset.remove_columns(['pos_tags', 'chunk_tags', 'id' ])
dataset = dataset.filter(lambda example: example["ner_tags"].count(7) == 0)
dataset = dataset.filter(lambda example: example["ner_tags"].count(8) == 0)
#datasets = datasets.rename_column("tokens", "words")
#datasets = datasets.rename_column("ner_tags", "ner")


dataset_as = load_dataset("ai4bharat/ner", 'as', use_auth_token=True)
dataset_bn = load_dataset("ai4bharat/ner", 'bn', use_auth_token=True)
dataset_gu = load_dataset("ai4bharat/ner", 'gu', use_auth_token=True)
dataset_hi = load_dataset("ai4bharat/ner", 'hi', use_auth_token=True)
dataset_kn = load_dataset("ai4bharat/ner", 'kn', use_auth_token=True)
dataset_ml = load_dataset("ai4bharat/ner", 'ml', use_auth_token=True)
dataset_mr = load_dataset("ai4bharat/ner", 'mr', use_auth_token=True)
dataset_or = load_dataset("ai4bharat/ner", 'or', use_auth_token=True)
dataset_pa = load_dataset("ai4bharat/ner", 'pa', use_auth_token=True)
dataset_ta = load_dataset("ai4bharat/ner", 'ta', use_auth_token=True)
dataset_te = load_dataset("ai4bharat/ner", 'te', use_auth_token=True)



# This method is adapted from the huggingface transformers run_ner.py example script
# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    text_column_name = "tokens"
    label_column_name = "ner_tags"
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        max_length=256,
        #padding=False,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


test_dataset = dataset["test"]
dataset_en = dataset.map(
    tokenize_and_align_labels,
    batched=True,
)

test_as = dataset_as['test'].map(tokenize_and_align_labels, batched=True,)
test_bn = dataset_bn['test'].map(tokenize_and_align_labels, batched=True,)
test_gu = dataset_gu['test'].map(tokenize_and_align_labels, batched=True,)
test_hi = dataset_hi['test'].map(tokenize_and_align_labels, batched=True,)
test_kn = dataset_kn['test'].map(tokenize_and_align_labels, batched=True,)
test_ml = dataset_ml['test'].map(tokenize_and_align_labels, batched=True,)
test_mr = dataset_mr['test'].map(tokenize_and_align_labels, batched=True,)
test_or = dataset_or['test'].map(tokenize_and_align_labels, batched=True,)
test_pa = dataset_pa['test'].map(tokenize_and_align_labels, batched=True,)
test_ta = dataset_ta['test'].map(tokenize_and_align_labels, batched=True,)
test_te = dataset_te['test'].map(tokenize_and_align_labels, batched=True,)



data_collator = DataCollatorForTokenClassification(tokenizer,)




# Metrics
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    label_list = id_2_label

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

training_args = TrainingArguments(
    f"callbacks_inB_ner_FT{args.batch_size}_{args.learning_rate}",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    #gradient_accumulation_steps =4,
    metric_for_best_model = 'eval_f1',
    greater_is_better = True,
    fp16=True,
)




print("########################### ZERO SHOT EVALUATION ###########################################")

def eval_test_lang(data_test, data_name):
  print("zero shot test performance for lang:   ", data_name )
  eval_trainer = Trainer(
    model=multitask_model.taskmodels_dict["ner"],
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    #args=TrainingArguments(output_dir="./eval_output", remove_unused_columns=False,),
    eval_dataset= data_test,
    )
  metric = eval_trainer.evaluate()
  print("the metric for language ", data_name , " is : ",  metric)
  wandb.log({"en-{}".format(data_name): metric})
  wandb.log({"en-{}-eval_loss".format(data_name): metric.get('eval_loss')})
  wandb.log({"en-{}-eval_precision".format(data_name): metric.get('eval_precision')})
  wandb.log({"en-{}-eval_recall".format(data_name): metric.get('eval_recall')})
  wandb.log({"en-{}-eval_f1".format(data_name): metric.get('eval_f1')})
  wandb.log({"en-{}-eval_accuracy".format(data_name): metric.get('eval_accuracy')})
  wandb.log({"en-{}-eval_runtime".format(data_name): metric.get('eval_runtime')})
  wandb.log({"en-{}-eval_samples_per_second".format(data_name): metric.get('eval_samples_per_second')})
  wandb.log({"en-{}-eval_steps_per_second".format(data_name): metric.get('eval_steps_per_second')})
  return

eval_test_lang(dataset_en['test'], "en_test" )
eval_test_lang(test_as, "as" )
eval_test_lang(test_bn, "bn" )
eval_test_lang(test_gu, "gu" )
eval_test_lang(test_hi, "hi" )
eval_test_lang(test_kn, "kn" )
eval_test_lang(test_ml, "ml" )
eval_test_lang(test_mr, "mr" )
eval_test_lang(test_or, "or" )
eval_test_lang(test_pa, "pa" )
eval_test_lang(test_ta, "ta" )
eval_test_lang(test_te, "te" )
eval_test_lang(dataset_en['validation'], "en_val" )



