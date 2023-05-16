import argparse
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoAdapterModel, AutoTokenizer,Trainer, TrainingArguments,set_seed
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers.adapters import PfeifferConfig, HoulsbyConfig, LoRAConfig, CompacterConfig, PrefixTuningConfig
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
import torch
import os
import wandb


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--adap_drop", default="noAD", help = "adap drop[AD | noAD]")
parser.add_argument('--adapter_type', default = "pfeiffer", help = "adapter type[houlsby|pfeiffer|lora|compacter|houlsbyparallel|pfeifferparallel |prefixtuning]")
args = parser.parse_args()

wandb.init(project="your_project_name", entity="your_entity" , name = f"{args.adapter_type}_{args.adap_drop}_{args.batch_size}" )
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#set_seed(args.seed)


def preprocess_function(data):
    data = data.map(lambda batch: tokenizer.encode_plus( batch['sentence1'], max_length= args.max_seq_length, pad_to_max_length = True, truncation=True))
    data = data.rename_column("label", "labels")
    data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return data


def amazon_mapper(example):
    if example["stars"] > 3:
        example["label"] = 1
    elif example["stars"] < 3:
        example["label"] = 0
    return {"sentence1": example["review_body"], "label": example["label"]}

def test_mapper(example):
    if example["LABEL"] == "Positive":
        example["LABEL"] = 1
    else:
        example["LABEL"] = 0
    return {"sentence1": example["INDIC REVIEW"], "label": example["LABEL"]}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

dataset = load_dataset("amazon_reviews_multi", "en")
dataset = dataset.filter(lambda example: example["stars"] != 3)
dataset = dataset.map(amazon_mapper, remove_columns=dataset['train'].column_names)
label_list = dataset['train'].unique('label')
print(len(label_list))


model_name = 'ai4bharat/IndicBERT-MLM-only'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoAdapterModel.from_pretrained(model_name, num_labels=len(label_list), use_auth_token=True)


if args.adapter_type == "houlsby":
  config_a = HoulsbyConfig()
  model.add_adapter("houlsby_adapter", config=config_a)
  model.add_classification_head("houlsby_adapter", num_labels=len(label_list))
  model.train_adapter("houlsby_adapter")
  adapter_name = "houlsby_adapter"

elif args.adapter_type == "pfeiffer":
  config_a = PfeifferConfig()
  model.add_adapter("pfeiffer_adapter", config=config_a)
  model.add_classification_head("pfeiffer_adapter", num_labels=len(label_list))
  model.train_adapter("pfeiffer_adapter")
  adapter_name = "pfeiffer_adapter"

elif args.adapter_type == "lora":
  config_a = LoRAConfig(r=8, alpha=16)
  model.add_adapter("lora_adapter", config=config_a)
  model.add_classification_head("lora_adapter", num_labels=len(label_list))
  model.train_adapter("lora_adapter")
  adapter_name = "lora_adapter"

elif args.adapter_type == "compacter":
  config_a = CompacterConfig()
  model.add_adapter("Compacter_adapter", config=config_a)
  model.add_classification_head("Compacter_adapter", num_labels=len(label_list))
  model.train_adapter("Compacter_adapter")
  adapter_name = "Compacter_adapter"

elif args.adapter_type == "houlsbyparallel":
  config_a = HoulsbyConfig(is_parallel = True)
  model.add_adapter("houlsby_P_adapter", config=config_a)
  model.add_classification_head("houlsby_P_adapter", num_labels=len(label_list))
  model.train_adapter("houlsby_P_adapter")
  adapter_name = "houlsby_P_adapter"

elif args.adapter_type == "pfeifferparallel":
  config_a = PfeifferConfig(is_parallel = True)
  model.add_adapter("pfeiffer_P_adapter", config=config_a)
  model.add_classification_head("pfeiffer_P_adapter", num_labels=len(label_list))
  model.train_adapter("pfeiffer_P_adapter")
  adapter_name = "pfeiffer_P_adapter"

elif args.adapter_type == "prefixtuning":
  config_a = PrefixTuningConfig(flat=False, prefix_length=30)
  model.add_adapter("prefix_adapter", config=config_a)
  model.add_classification_head("prefix_adapter", num_labels=len(label_list))
  model.train_adapter("prefix_adapter")
  adapter_name = "prefix_adapter"


print(model)

for name, param in model.named_parameters():
     print(name, param.requires_grad)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
wandb.log({"total_parameter": pytorch_total_params})


metric = load_metric('glue', 'sst2')


#dataset['train'] = dataset['train'].shard(num_shards=20, index=0)
#dataset['validation'] = dataset['validation'].shard(num_shards=100, index=0)
dataset_en = preprocess_function(dataset)
print(dataset_en)

def test_preprocess(test_data):
    test_data = test_data.map(test_mapper)
    test_data = preprocess_function(test_data["test"])
    return test_data


test_as = load_dataset("ai4bharat/IndicSentiment", "translation-as", use_auth_token=True)
test_as = test_preprocess(test_as)

test_bn = load_dataset("ai4bharat/IndicSentiment", "translation-bn", use_auth_token=True)
test_bn = test_preprocess(test_bn)

test_gu = load_dataset("ai4bharat/IndicSentiment", "translation-gu", use_auth_token=True)
test_gu = test_preprocess(test_gu)

test_hi = load_dataset("ai4bharat/IndicSentiment", "translation-hi", use_auth_token=True)
test_hi = test_preprocess(test_hi)

test_kn = load_dataset("ai4bharat/IndicSentiment", "translation-kn", use_auth_token=True)
test_kn = test_preprocess(test_kn)

test_ml = load_dataset("ai4bharat/IndicSentiment", "translation-ml", use_auth_token=True)
test_ml = test_preprocess(test_ml)

test_mr = load_dataset("ai4bharat/IndicSentiment", "translation-mr", use_auth_token=True)
test_mr = test_preprocess(test_mr)

test_or = load_dataset("ai4bharat/IndicSentiment", "translation-or", use_auth_token=True)
test_or = test_preprocess(test_or)

test_pa = load_dataset("ai4bharat/IndicSentiment", "translation-pa", use_auth_token=True)
test_pa = test_preprocess(test_pa)

test_ta = load_dataset("ai4bharat/IndicSentiment", "translation-ta", use_auth_token=True)
test_ta = test_preprocess(test_ta)

test_te = load_dataset("ai4bharat/IndicSentiment", "translation-te", use_auth_token=True)
test_te = test_preprocess(test_te)

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
    f"callbacks_inB_AdapterDrop{args.batch_size}_{args.adapter_type}_{args.adap_drop}_STA",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    warmup_steps= 2000,
    num_train_epochs=50,
    evaluation_strategy = "epoch", #IntervalStrategy.STEPS,
    save_strategy = "epoch", #IntervalStrategy.STEPS,
    # eval_steps = 1000,
    # save_steps =1000 ,
    save_total_limit = 3,
    # logging_steps=1000,
    metric_for_best_model = 'eval_accuracy',
    greater_is_better = True,
    overwrite_output_dir=True,
    remove_unused_columns=False,
    #dataloader_num_workers= 4,
    load_best_model_at_end=True,
    fp16=True,
    seed = args.seed,
    )

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_en["train"],
    eval_dataset=dataset_en["validation"],
    #tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience= 3 )]
)
if args.adap_drop == "AD":
  trainer.add_callback(AdapterDropTrainerCallback())

trainer.train()
trainer.evaluate()
model.save_adapter(f"/sentiment_its_ok", "pfeiffer_adapter")

def eval_test_lang(data_test, data_name):
  print("zero shot test performance for lang:   ", data_name )
  eval_trainer = AdapterTrainer(
    model=model,
    args=training_args, #TrainingArguments(output_dir="./eval_output", remove_unused_columns=False,),
    eval_dataset= data_test,
    compute_metrics=compute_metrics,
    )
  metric = eval_trainer.evaluate()
  print("the metric for language ", data_name , " is : ",  metric)
  wandb.log({"en-{}".format(data_name): metric})
  wandb.log({"en-{}-eval_loss".format(data_name): metric.get('eval_loss')})
  wandb.log({"en-{}-eval_accuracy".format(data_name): metric.get('eval_accuracy')})
  wandb.log({"en-{}-eval_runtime".format(data_name): metric.get('eval_runtime')})
  wandb.log({"en-{}-eval_steps_per_second".format(data_name): metric.get('eval_steps_per_second')})
  return

eval_test_lang(dataset_en['test'], "en" )
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
eval_test_lang(dataset_en['validation'], "en-val" )
