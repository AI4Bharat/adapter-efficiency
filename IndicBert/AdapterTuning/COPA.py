import argparse

import numpy as np
from datasets import load_dataset, load_metric
from transformers import (AutoModelForMultipleChoice, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
from transformers import EarlyStoppingCallback, IntervalStrategy
import os
from transformers.adapters import PfeifferConfig, HoulsbyConfig, LoRAConfig, CompacterConfig, PrefixTuningConfig
#os.environ["WANDB_DISABLED"] = "true"
import wandb


parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, default="siqa")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default= 3e-5)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--reduction_factor", type=int, default=16)
parser.add_argument("--adap_drop", default="noAD", help = "adap drop[AD | noAD]")
parser.add_argument('--adapter_type', default = "houlsby", help = "adapter type[houlsby|pfeiffer|lora|compacter|houlsbyparallel|pfeifferparallel |prefixtuning]")

args = parser.parse_args()

set_seed(args.seed)

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

tokenizer = AutoTokenizer.from_pretrained("/nlsasfs/home/ai4bharat/nandinim/nandini/new_finetune/copa/tok_InBert_mlm_only/", use_auth_token=True)
model = AutoModelForMultipleChoice.from_pretrained("/nlsasfs/home/ai4bharat/nandinim/nandini/new_finetune/copa/model_InBert_mlm_only_copa/", use_auth_token=True)


if args.adapter_type == "houlsby":
  config_a = HoulsbyConfig(reduction_factor = args.reduction_factor)
  model.add_adapter("houlsby_adapter", config=config_a)
  model.train_adapter("houlsby_adapter")
  adapter_name = "houlsby_adapter"

elif args.adapter_type == "pfeiffer":
  config_a = PfeifferConfig(reduction_factor = args.reduction_factor)
  model.add_adapter("pfeiffer_adapter", config=config_a)
  model.train_adapter("pfeiffer_adapter")
  adapter_name = "pfeiffer_adapter"

elif args.adapter_type == "lora":
  config_a = LoRAConfig(r=8, alpha=16)
  model.add_adapter("lora_adapter", config=config_a)
  model.train_adapter("lora_adapter")
  adapter_name = "lora_adapter"

elif args.adapter_type == "compacter":
  config_a = CompacterConfig()
  model.add_adapter("Compacter_adapter", config=config_a)
  model.train_adapter("Compacter_adapter")
  adapter_name = "Compacter_adapter"

elif args.adapter_type == "houlsbyparallel":
  config_a = HoulsbyConfig(is_parallel = True)
  model.add_adapter("houlsby_P_adapter", config=config_a)
  model.train_adapter("houlsby_P_adapter")
  adapter_name = "houlsby_P_adapter"

elif args.adapter_type == "pfeifferparallel":
  config_a = PfeifferConfig(is_parallel = True)
  model.add_adapter("pfeiffer_P_adapter", config=config_a)
  model.train_adapter("pfeiffer_P_adapter")
  adapter_name = "pfeiffer_P_adapter"

elif args.adapter_type == "prefixtuning":
  config_a = PrefixTuningConfig(flat=False, prefix_length=30)
  model.add_adapter("prefix_adapter", config=config_a)
  model.train_adapter("prefix_adapter")
  adapter_name = "prefix_adapter"




pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# wandb.log({"total_parameter": pytorch_total_params})


def convert_label_to_int(example):
    # for siqa
    example["label"] = int(example["label"]) - 1
    return example

dataset['train'] = dataset['train'].map(convert_label_to_int)
dataset['validation'] = dataset['validation'].map(convert_label_to_int)

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

import numpy as np
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction, TrainerCallback

class AdapterDropTrainerCallback(TrainerCallback):
  def on_step_begin(self, args, state, control, **kwargs):
    skip_layers = list(range(np.random.randint(0, 11)))
    kwargs['model'].set_active_adapters(adapter_name, skip_layers=skip_layers)

  def on_evaluate(self, args, state, control, **kwargs):
    # Deactivate skipping layers during evaluation (otherwise it would use the
    # previous randomly chosen skip_layers and thus yield results not comparable
    # across different epochs)
    kwargs['model'].set_active_adapters(adapter_name, skip_layers=None)


training_args = TrainingArguments(
    f"callback_{args.batch_size}_{args.adapter_type}_{args.adap_drop}_its_ok_STA",
    num_train_epochs=50,
    evaluation_strategy ="epoch", #IntervalStrategy.STEPS,
    save_strategy = "epoch", #IntervalStrategy.STEPS,
    # eval_steps = 1000,
    # save_steps =1000 ,
    # logging_steps=1000,
    metric_for_best_model = 'eval_accuracy',
    greater_is_better = True,
    overwrite_output_dir=True,
    remove_unused_columns=False,
    #dataloader_num_workers= 4,
    load_best_model_at_end=True,
    fp16=True,
    seed = 42,
    save_total_limit=3,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    warmup_steps= 2000,
)

# earlystoppingcallback = EarlyStoppingCallback(early_stopping_patience=2)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience= 3 )]
    # callbacks=[earlystoppingcallback]
)
#model.load_state_dict(torch.load("/nlsasfs/home/ai4bharat/nandinim/nandini/new_adaptertune/copa/callback_64_houlsby_3/checkpoint-17000/pytorch_model.bin" ))

#trainer.add_callback(AdapterDropTrainerCallback())
if args.adap_drop == "AD":
  trainer.add_callback(AdapterDropTrainerCallback())

trainer.train()

trainer.evaluate()
model.save_adapter(f"/nlsasfs/home/ai4bharat/nandinim/nandini/new_adaptertune/adapter_fusion/copa_its_ok", "pfeiffer_adapter")

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
dataset_gom = load_dataset("ai4bharat/IndicCOPA",'translation-gom', use_auth_token=True)
dataset_mai = load_dataset("ai4bharat/IndicCOPA",'translation-mai', use_auth_token=True)
dataset_ne = load_dataset("ai4bharat/IndicCOPA",'translation-ne', use_auth_token=True)
dataset_sa = load_dataset("ai4bharat/IndicCOPA",'translation-sa', use_auth_token=True)
dataset_sat = load_dataset("ai4bharat/IndicCOPA",'translation-sat', use_auth_token=True)
dataset_sd = load_dataset("ai4bharat/IndicCOPA",'translation-ur', use_auth_token=True)


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


def zero_val(dataset, data_name):
    
    test_dataset = dataset['test'].map(
        preprocess_function_indic,
        batched=True,
        # remove_columns=dataset['test'].column_names,
    )
    test_dataset = test_dataset.remove_columns(['premise', 'choice1', 'choice2', 'question', 'idx', 'changed' ])
    print(f"Length of Test dataset: {len(test_dataset)}", test_dataset)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
    )

    results = trainer.predict(test_dataset)
    print(f"Results for  dataset: {results.metrics}")
#     wandb.log({"en-{}".format(data_name): results.metrics})
#     wandb.log({"en-{}-test_loss".format(data_name): results.metrics.get('test_loss')})
#     wandb.log({"en-{}-test_eval_accuracy".format(data_name): results.metrics.get('test_eval_accuracy')})
#     wandb.log({"en-{}-test_runtime".format(data_name): results.metrics.get('test_runtime')})
#     wandb.log({"en-{}-test_samples_per_second".format(data_name): results.metrics.get('test_samples_per_second')})
#     wandb.log({"en-{}-test_steps_per_second".format(data_name): results.metrics.get('test_steps_per_second')})


    

#dataset = load_dataset("ai4bharat/IndicXCOPA", f"{args.eval_data}", use_auth_token=True)
#zero_val(dataset_en, 'en')
zero_val(dataset_sat, 'sat')
zero_val(dataset_sa, 'sa')
zero_val(dataset_gom, 'gom')
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
zero_val(dataset_gom, 'gom')
zero_val(dataset_mai, 'mai')
zero_val(dataset_ne, 'ne')
zero_val(dataset_sa, 'sa')
zero_val(dataset_sat, 'sat')
zero_val(dataset_sd, 'sd')







