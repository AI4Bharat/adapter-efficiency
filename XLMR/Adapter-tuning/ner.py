from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
from transformers import AutoConfig, AutoAdapterModel
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch import nn
import copy
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from transformers.adapters import PfeifferConfig, HoulsbyConfig, LoRAConfig, CompacterConfig, PrefixTuningConfig
from datasets import load_metric
import numpy as np
from transformers import EarlyStoppingCallback, IntervalStrategy
import argparse
import wandb
#The labels for the NER task and the dictionaries to map the to ids or 
#the other way around
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--max_seq_length", type=int, default=256)
parser.add_argument("--reduction_factor", type=int, default=16)
parser.add_argument("--adap_drop", default="noAD", help = "adapter type[AD | noAD]")
parser.add_argument("--run_name", default="its_ok_")
parser.add_argument("--model_name", default="xlmr-b", help= "model type [xlmr-b | xlmr-l| indicbert]")
parser.add_argument('--adapter_type', default = "houlsby", help = "adapter type[houlsby|pfeiffer|lora|compacter|houlsbyparallel|pfeifferparallel |prefixtuning]")

args = parser.parse_args()
labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
id_2_label = {id_: label for id_, label in enumerate(labels)}
label_2_id = {label: id_ for id_, label in enumerate(labels)}

if args.model_name == "indicbert":
    wandb.init(project="Indicbert_mlm_only_xnli", entity="nandinimundra", name = f"{args.model_name}_{args.adapter_type}_{args.adap_drop}_{args.batch_size}_{args.learning_rate}")
    model_name = 'ai4bharat/IndicBERT-MLM-only'
    tokenizer = AutoTokenizer.from_pretrained("/nlsasfs/home/ai4bharat/nandinim/nandini/new_adaptertune/xnli/tok_InBert_mlm_only/", use_auth_token=True)
    config = AutoConfig.from_pretrained(
        "/nlsasfs/home/ai4bharat/nandinim/nandini/new_adaptertune/xnli/config_InBert_mlm_only_xnli_adap/",
        num_labels=3,
        id2label={ 0: "entailment", 1: "neutral", 2: "contradiction"},
        use_auth_token=True,
    )
    model = AutoAdapterModel.from_pretrained(
        "/nlsasfs/home/ai4bharat/nandinim/nandini/new_adaptertune/xnli/model_InBert_mlm_only_xnli_adap/",
        config=config,
        use_auth_token=True,
    )

elif args.model_name == "xlmr-b":
    wandb.init(project="xlmr_base_ner", entity="nandinimundra", name = f"{args.model_name}_{args.adapter_type}_{args.run_name}_{args.batch_size}_{args.adap_drop}")
    model_name = 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_auth_token=True)
    config = AutoConfig.from_pretrained(model_name, num_labels=len(labels), label2id=label_2_id, id2label=id_2_label)
    model = AutoAdapterModel.from_pretrained(model_name, config=config)

elif args.model_name == "xlmr-l":
    wandb.init(project="xlmr_large_ner", entity="nandinimundra", name = f"{args.model_name}_{args.adapter_type}_{args.run_name}_{args.batch_size}_{args.adap_drop}")
    model_name = 'xlm-roberta-large'
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large", use_auth_token=True)
    config = AutoConfig.from_pretrained(model_name, num_labels=len(labels), label2id=label_2_id, id2label=id_2_label)
    model = AutoAdapterModel.from_pretrained(model_name, config=config)


if args.adapter_type == "houlsby":
  config_a = HoulsbyConfig(reduction_factor = args.reduction_factor)
  model.add_adapter("houlsby_adapter", config=config_a)
  model.add_tagging_head("houlsby_adapter", num_labels=len(labels),  id2label=id_2_label)
  model.train_adapter("houlsby_adapter")
  adapter_name = "houlsby_adapter"

elif args.adapter_type == "pfeiffer":
    config_a = PfeifferConfig(reduction_factor = args.reduction_factor)
    model.add_adapter("pfeiffer_adapter", config=config_a)
    model.add_tagging_head("pfeiffer_adapter", num_labels=len(labels),  id2label=id_2_label)
    model.train_adapter("pfeiffer_adapter")
    adapter_name = "pfeiffer_adapter"

elif args.adapter_type == "lora":
  config_a = LoRAConfig(r=8, alpha=16)
  model.add_adapter("lora_adapter", config=config_a)
  model.add_tagging_head("lora_adapter", num_labels=len(labels),  id2label=id_2_label)
  model.train_adapter("lora_adapter")
  adapter_name = "lora_adapter"

elif args.adapter_type == "compacter":
  config_a = CompacterConfig()
  model.add_adapter("Compacter_adapter", config=config_a)
  model.add_tagging_head("Compacter_adapter", num_labels=len(labels),  id2label=id_2_label)
  model.train_adapter("Compacter_adapter")
  adapter_name = "Compacter_adapter"

elif args.adapter_type == "houlsbyparallel":
  config_a = HoulsbyConfig(is_parallel = True)
  model.add_adapter("houlsby_P_adapter", config=config_a)
  model.add_tagging_head("houlsby_P_adapter", num_labels=len(labels),  id2label=id_2_label)
  model.train_adapter("houlsby_P_adapter")
  adapter_name = "houlsby_P_adapter"

elif args.adapter_type == "pfeifferparallel":
  config_a = PfeifferConfig(is_parallel = True)
  model.add_adapter("pfeiffer_P_adapter", config=config_a)
  model.add_tagging_head("pfeiffer_P_adapter", num_labels=len(labels),  id2label=id_2_label)
  model.train_adapter("pfeiffer_P_adapter")
  adapter_name = "pfeiffer_P_adapter"

elif args.adapter_type == "prefixtuning":
  config_a = PrefixTuningConfig(flat=False, prefix_length=30)
  model.add_adapter("prefix_adapter", config=config_a)
  model.add_tagging_head("prefix_adapter", num_labels=len(labels),  id2label=id_2_label)
  model.train_adapter("prefix_adapter")
  adapter_name = "prefix_adapter"

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
wandb.log({"no_of_parameter": pytorch_total_params})

#ar, bg, de, el, es, fr, hi, ru, sw, th, tr, ur, vi, zh
datasets = load_dataset('wikiann', 'en')
datasets = datasets.shuffle(seed=42)

dataset_ar = load_dataset('wikiann', 'ar')
dataset_bg = load_dataset('wikiann', 'bg')
dataset_de = load_dataset('wikiann', 'de')
dataset_el = load_dataset('wikiann', 'el')
dataset_es = load_dataset('wikiann', 'es')
dataset_fr = load_dataset('wikiann', 'fr')
dataset_hi = load_dataset('wikiann', 'hi')
dataset_ru = load_dataset('wikiann', 'ru')
dataset_sw = load_dataset('wikiann', 'sw')
dataset_th = load_dataset('wikiann', 'th')
dataset_tr = load_dataset('wikiann', 'tr')
dataset_ur = load_dataset('wikiann', 'ur')
dataset_vi = load_dataset('wikiann', 'vi')
dataset_zh = load_dataset('wikiann', 'zh')



training_args = TrainingArguments(
    f"callbacks_inB_ner_{args.adapter_type}_{args.run_name}_{args.adap_drop}_{args.learning_rate}_{args.batch_size}_{args.model_name}",
    evaluation_strategy = "epoch", #IntervalStrategy.STEPS,
    save_strategy = "epoch", #IntervalStrategy.STEPS,
    #gradient_accumulation_steps =args.grad_acc,
    num_train_epochs=50,
    save_total_limit = 4,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    warmup_steps= 2000,
    #warmup_ratio=args.warmup_ratio,
    do_predict=True,
    #output_dir="ner_models/xlmr/",
    load_best_model_at_end=True,
    metric_for_best_model = 'eval_f1',
    greater_is_better = True,
    fp16=True,
)

# This method is adapted from the huggingface transformers run_ner.py example script 
# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    text_column_name = "tokens"
    label_column_name = "ner_tags"
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=False,
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


test_dataset = datasets["test"]
dataset_tok = datasets.map(
    tokenize_and_align_labels,
    batched=True,
)


test_ar = dataset_ar['test'].map(tokenize_and_align_labels, batched=True,)
test_bg = dataset_bg['test'].map(tokenize_and_align_labels, batched=True,)
test_de = dataset_de['test'].map(tokenize_and_align_labels, batched=True,)
test_el = dataset_el['test'].map(tokenize_and_align_labels, batched=True,)
test_es = dataset_es['test'].map(tokenize_and_align_labels, batched=True,)
test_fr = dataset_fr['test'].map(tokenize_and_align_labels, batched=True,)
test_hi = dataset_hi['test'].map(tokenize_and_align_labels, batched=True,)
test_ru = dataset_ru['test'].map(tokenize_and_align_labels, batched=True,)
test_sw = dataset_sw['test'].map(tokenize_and_align_labels, batched=True,)
test_th = dataset_th['test'].map(tokenize_and_align_labels, batched=True,)
test_tr = dataset_tr['test'].map(tokenize_and_align_labels, batched=True,)
test_ur = dataset_ur['test'].map(tokenize_and_align_labels, batched=True,)
test_vi = dataset_vi['test'].map(tokenize_and_align_labels, batched=True,)
test_zh = dataset_zh['test'].map(tokenize_and_align_labels, batched=True,)

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

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_tok['train'],
    eval_dataset=dataset_tok['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience= 3 )]
)
if args.adap_drop == "AD":
  trainer.add_callback(AdapterDropTrainerCallback())

print(trainer.train())
print(trainer.evaluate())

print("########################### ZERO SHOT EVALUATION ###########################################")

def eval_test_lang(data_test, data_name):
  print("zero shot test performance for lang:   ", data_name )
  eval_trainer = AdapterTrainer(
    model=model,
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

eval_test_lang(dataset_tok['test'], "en" )
eval_test_lang(test_ar, 'ar')
eval_test_lang(test_bg, 'bg')
eval_test_lang(test_de, 'de')
eval_test_lang(test_el, 'el')
eval_test_lang(test_es, 'es')
eval_test_lang(test_fr, 'fr')
eval_test_lang(test_hi, 'hi')
eval_test_lang(test_ru, 'ru')
eval_test_lang(test_sw, 'sw')
eval_test_lang(test_th, 'th')
eval_test_lang(test_tr, 'tr')
eval_test_lang(test_ur, 'ur')
eval_test_lang(test_vi, 'vi')
eval_test_lang(test_zh, 'zh')



