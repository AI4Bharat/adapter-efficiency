import argparse
from torch import nn
import numpy as np
from datasets import load_dataset, load_metric
from transformers import (AutoModelForMultipleChoice, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
from transformers import EarlyStoppingCallback, IntervalStrategy
import os, transformers

import wandb
#os.environ["WANDB_DISABLED"] = "true"







parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, default="siqa")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_path", type=str, default="callbacks_MTL_xnli_pref_ep_10_8__32_4e-05_7")
parser.add_argument("--checkpont_p", type=str, default="checkpoint-69861")
args = parser.parse_args()
wandb.init(project="indicbert_mlm_only_copa",  entity="your_entity" , name = f"last_please_{args.model_path}_{args.checkpont_p}" )

set_seed(args.seed)
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
multitask_model.load_state_dict(torch.load(f"/new_finetune/MTL/{args.model_path}/{args.checkpont_p}/pytorch_model.bin" ))
@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """
    #def __init__(self, tokenizer):
     # self.tokenizer = tokenizer


    tokenizer: PreTrainedTokenizerBase
    #padding: Union[bool, str, PaddingStrategy] = True
    #max_length: Optional[int] = None
    #pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        #print("in call of data collaot : ", features[0].keys())
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        #print("flattened feature in data, ", flattened_features)
        flattened_features = sum(flattened_features, [])

        #print("flattened feature in data AFTER PROCESSING, ", flattened_features)
        
        batch = self.tokenizer.pad(
            flattened_features,
            #padding= True,
            padding= 'max_length',
            max_length= 256,
            #pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            #truncation=True ,
            
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        #print("in data collator " , batch)
        return batch

def preprocess_function(examples):

    if args.task_name == "siqa":
        ending_names = [f"answer{i}" for i in "ABC"]
        context_name = "context"
        question_header_name = "question"
    elif args.task_name == "xcopa":
        ending_names = [f"choice{i}" for i in "12"]
        context_name = "premise"
        question_header_name = "question"

    first_sentences = [
        [context] * len(ending_names) for context in examples[context_name]
    ]
    question_headers = examples[question_header_name]
    if args.task_name == "xcopa" or args.task_name == "siqa":
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names]
            for i, header in enumerate(question_headers)
        ]
    else:
        # remove {header} as our dataset has not question headers
        second_sentences = [
            [f"{examples[end][i]}" for end in ending_names]
            for i, header in enumerate(question_headers)
        ]

    # Flatten out
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length= 256,
        pad_to_max_length = True,
        # padding="max_length",
    )
    # Un-flatten
    return {
        k: [
            v[i : i + len(ending_names)]
            for i in range(0, len(v), len(ending_names))
        ]
        for k, v in tokenized_examples.items()
    }

# Metric
def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    m = {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}
    return {f"eval_{k}" if "eval_" not in k else k: v for k, v in m.items()}

dataset = load_dataset("social_i_qa")

#dataset['train'] = dataset['train'].shard(num_shards=200, index=0)
#dataset['validation'] = dataset['validation'].shard(num_shards=20, index=0)

label_list = dataset['train'].unique("label")
label_list.sort()  # Let's sort it for determinism
label_to_id = {v: i for i, v in enumerate(label_list)}


def convert_label_to_int(example):
    # for siqa
    example["label"] = int(example["label"]) - 1
    return example

dataset['train'] = dataset['train'].map(convert_label_to_int)
dataset['validation'] = dataset['validation'].map(convert_label_to_int)
print(dataset)


train_dataset = dataset["train"].map(
    preprocess_function,
    batched=True,
    # remove_columns=dataset["train"].column_names,
)
print(f"Length of Training dataset: {len(train_dataset)}")
train_dataset = train_dataset.remove_columns(['context','question', 'answerA', 'answerC', 'answerB' ])

validation_dataset = dataset['validation'].map(
    preprocess_function,
    batched=True,
    # remove_columns=dataset['validation'].column_names,
)
validation_dataset = validation_dataset.remove_columns(['context','question', 'answerA', 'answerC', 'answerB' ])
print(f"Length of Validation dataset: {len(validation_dataset)}")


training_args = TrainingArguments(
    f"callbacks_inB_Senti_FT{args.batch_size}_{args.learning_rate}",
    num_train_epochs=50,
    evaluation_strategy = IntervalStrategy.STEPS,
    save_strategy = IntervalStrategy.STEPS,
    eval_steps = 1000,
    save_steps =1000 ,
    logging_steps=1000,
    metric_for_best_model = 'eval_accuracy',
    greater_is_better = True,
    overwrite_output_dir=True,
    remove_unused_columns=False,
    #dataloader_num_workers= 4,
    load_best_model_at_end=True,
    fp16=True,
    seed = 42,
    save_total_limit=5,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    #warmup_ratio=args.warmup_ratio,
)

# earlystoppingcallback = EarlyStoppingCallback(early_stopping_patience=2)

dataset_as = load_dataset("ai4bharat/IndicCOPA",'translation-as', use_auth_token=True)
dataset_bn = load_dataset("ai4bharat/IndicCOPA",'translation-bn', use_auth_token=True)
dataset_gu = load_dataset("ai4bharat/IndicCOPA",'translation-gu', use_auth_token=True)
dataset_hi = load_dataset("ai4bharat/IndicCOPA",'translation-hi', use_auth_token=True)
dataset_kn = load_dataset("ai4bharat/IndicCOPA",'translation-kn', use_auth_token=True)
dataset_ml = load_dataset("ai4bharat/IndicCOPA",'translation-ml', use_auth_token=True)
dataset_mr = load_dataset("ai4bharat/IndicCOPA",'translation-mr', use_auth_token=True)
dataset_or = load_dataset("ai4bharat/IndicCOPA",'translation-or', use_auth_token=True)
dataset_pa = load_dataset("ai4bharat/IndicCOPA",'translation-pa', use_auth_token=True)
dataset_ta = load_dataset("ai4bharat/IndicCOPA",'translation-ta', use_auth_token=True)
dataset_te = load_dataset("ai4bharat/IndicCOPA",'translation-te', use_auth_token=True)
def preprocess_function_indic(examples):
    
    ending_names = [f"choice{i}" for i in "12"]
    context_name = "premise"
    question_header_name = "question"

    first_sentences = [
        [context] * len(ending_names) for context in examples[context_name]
    ]
    question_headers = examples[question_header_name]
    second_sentences = [
          [f"{header} {examples[end][i]}" for end in ending_names]
          for i, header in enumerate(question_headers)
      ]

    # Flatten out

    first_sentences = sum(first_sentences, [])

    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        padding=True,
        max_length= 256,
        # padding="max_length",
    )

    # Un-flatten    
    return {
        k: [
            v[i : i + len(ending_names)]
            for i in range(0, len(v), len(ending_names))
        ]
        for k, v in tokenized_examples.items()
        }


def zero_val_en(dataset, data_name):
    
    print(f"Length of Test dataset: {len(dataset)}", dataset)

    trainer = Trainer(
        model=multitask_model.taskmodels_dict["copa"],
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
    )

    results = trainer.predict(dataset)
    print(f"Results for  dataset: {results.metrics}")
    wandb.log({"en-{}".format(data_name): results.metrics})
    wandb.log({"en-{}-test_loss".format(data_name): results.metrics.get('test_loss')})
    wandb.log({"en-{}-test_eval_accuracy".format(data_name): results.metrics.get('test_eval_accuracy')})
    wandb.log({"en-{}-test_runtime".format(data_name): results.metrics.get('test_runtime')})
    wandb.log({"en-{}-test_samples_per_second".format(data_name): results.metrics.get('test_samples_per_second')})
    wandb.log({"en-{}-test_steps_per_second".format(data_name): results.metrics.get('test_steps_per_second')})


zero_val_en(validation_dataset, "en-val")






def zero_val(dataset, data_name):
    
    test_dataset = dataset['test'].map(
        preprocess_function_indic,
        batched=True,
        # remove_columns=dataset['test'].column_names,
    )
    test_dataset = test_dataset.remove_columns(['premise', 'choice1', 'choice2', 'question', 'idx', 'changed' ])
    print(f"Length of Test dataset: {len(test_dataset)}", test_dataset)

    trainer = Trainer(
        model=multitask_model.taskmodels_dict["copa"],
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
    )

    results = trainer.predict(test_dataset)
    print(f"Results for  dataset: {results.metrics}")
    wandb.log({"en-{}".format(data_name): results.metrics})
    wandb.log({"en-{}-test_loss".format(data_name): results.metrics.get('test_loss')})
    wandb.log({"en-{}-test_eval_accuracy".format(data_name): results.metrics.get('test_eval_accuracy')})
    wandb.log({"en-{}-test_runtime".format(data_name): results.metrics.get('test_runtime')})
    wandb.log({"en-{}-test_samples_per_second".format(data_name): results.metrics.get('test_samples_per_second')})
    wandb.log({"en-{}-test_steps_per_second".format(data_name): results.metrics.get('test_steps_per_second')})


    

#dataset = load_dataset("ai4bharat/IndicXCOPA", f"{args.eval_data}", use_auth_token=True)
#zero_val(dataset_en, 'en')
zero_val(dataset_as, 'as')
zero_val(dataset_bn, 'bn')
zero_val(dataset_gu, 'gu')
zero_val(dataset_hi, 'hi')
zero_val(dataset_kn, 'kn')
zero_val(dataset_ml, 'ml')
zero_val(dataset_mr, 'mr')
zero_val(dataset_or, 'or')
zero_val(dataset_pa, 'pa')
zero_val(dataset_ta, 'ta')
zero_val(dataset_te, 'te')

