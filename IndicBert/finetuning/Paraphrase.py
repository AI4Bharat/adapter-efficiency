import argparse

import numpy as np
from datasets import load_dataset, load_metric
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments,
                          set_seed)
from transformers import EarlyStoppingCallback, IntervalStrategy
import os
import wandb
#os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=0.1)

args = parser.parse_args()
wandb.init(project="indicbert_mlm_only_paraphrase", entity="nandinimundra" , name = f"FT_its_ok_{args.batch_size}_{args.learning_rate}" )


def preprocess(dataset):
  dataset = dataset.map(lambda batch: tokenizer.encode_plus( batch['sentence1'], batch['sentence2'], max_length= 256, pad_to_max_length = True, truncation=True))
  dataset = dataset.rename_column("label", "labels")
  dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
  return dataset

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)



dataset_en = load_dataset("paws-x", "en")
#dataset_en['train'] = dataset_en['train'].shard(num_shards=200, index=0)
#dataset_en['validation'] = dataset_en['validation'].shard(num_shards=200, index=0)


label_list = dataset_en["train"].features["label"].names
metric = load_metric('glue', 'mnli')

model_name = 'ai4bharat/IndicBERT-MLM-only'
tokenizer = AutoTokenizer.from_pretrained("/nlsasfs/home/ai4bharat/nandinim/nandini/new_finetune/paraphrase/tok_InBert_mlm_only/", use_auth_token=True)
model = AutoModelForSequenceClassification.from_pretrained("/nlsasfs/home/ai4bharat/nandinim/nandini/new_finetune/paraphrase/model_InBert_mlm_only_para/", num_labels=len(label_list), use_auth_token=True)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
wandb.log({"total_parameter": pytorch_total_params})

dataset_en_tok = preprocess(dataset_en)


training_args = TrainingArguments(
    f"callbacks_inB_Senti_FT{args.batch_size}_{args.learning_rate}_its_ok",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    warmup_steps= 2000,
    num_train_epochs=50,
    evaluation_strategy ="epoch", #IntervalStrategy.STEPS,
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
    


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_en_tok['train'],
    eval_dataset=dataset_en_tok['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience= 3 )]
)
trainer.train()

trainer.evaluate()

def eval_test_lang(data_test, data_name):
  print("zero shot test performance for lang:   ", data_name )
  data_test = preprocess(data_test)
  eval_trainer = Trainer(
    model=model,
    args= training_args , #TrainingArguments(output_dir="./eval_output", remove_unused_columns=False,),
    eval_dataset= data_test,
    compute_metrics=compute_metrics,
    )
  metric = eval_trainer.evaluate()
  print("the metric for language ", data_name , " is : ",  metric)
  wandb.log({"en-{}".format(data_name): metric})
  wandb.log({"en-{}-eval_loss".format(data_name): metric.get('eval_loss')})
  wandb.log({"en-{}-eval_accuracy".format(data_name): metric.get('eval_accuracy')})
  wandb.log({"en-{}-eval_runtime".format(data_name): metric.get('eval_runtime')})
  wandb.log({"en-{}-eval_samples_per_second".format(data_name): metric.get('eval_samples_per_second')})
  wandb.log({"en-{}-eval_steps_per_second".format(data_name): metric.get('eval_steps_per_second')})
  
  return

test_as = load_dataset("ai4bharat/IndicXParaphrase", 'as', use_auth_token=True)
test_bn = load_dataset("ai4bharat/IndicXParaphrase", 'bn', use_auth_token=True)
test_gu = load_dataset("ai4bharat/IndicXParaphrase", 'gu', use_auth_token=True)
test_hi = load_dataset("ai4bharat/IndicXParaphrase", 'hi', use_auth_token=True)
test_kn = load_dataset("ai4bharat/IndicXParaphrase", 'kn', use_auth_token=True)
test_ml = load_dataset("ai4bharat/IndicXParaphrase", 'ml', use_auth_token=True)
test_mr = load_dataset("ai4bharat/IndicXParaphrase", 'mr', use_auth_token=True)
test_or = load_dataset("ai4bharat/IndicXParaphrase", 'or', use_auth_token=True)
test_pa = load_dataset("ai4bharat/IndicXParaphrase", 'pa', use_auth_token=True)
#test_ta = load_dataset("ai4bharat/IndicXParaphrase", 'ta', use_auth_token=True)
test_te = load_dataset("ai4bharat/IndicXParaphrase", 'te', use_auth_token=True)


eval_test_lang(test_as['test'], "as" )
eval_test_lang(test_bn['test'], "bn" )
eval_test_lang(test_gu['test'], "gu" )
eval_test_lang(test_hi['test'], "hi" )
eval_test_lang(test_kn['test'], "kn" )
eval_test_lang(test_ml['test'], "ml" )
eval_test_lang(test_mr['test'], "mr" )
eval_test_lang(test_or['test'], "or" )
eval_test_lang(test_pa['test'], "pa" )
#eval_test_lang(test_ta, "ta" )
eval_test_lang(test_te['test'], "te" )



def eval_test_lang_2(data_test, data_name):
  print("zero shot test performance for lang:   ", data_name )
  #data_test = preprocess(data_test)
  eval_trainer = Trainer(
    model=model,
    args= training_args , #TrainingArguments(output_dir="./eval_output", remove_unused_columns=False,),
    eval_dataset= data_test,
    compute_metrics=compute_metrics,
    )
  metric = eval_trainer.evaluate()
  print("the metric for language ", data_name , " is : ",  metric)
  wandb.log({"en-{}".format(data_name): metric})
  wandb.log({"en-{}-eval_loss".format(data_name): metric.get('eval_loss')})
  wandb.log({"en-{}-eval_accuracy".format(data_name): metric.get('eval_accuracy')})
  wandb.log({"en-{}-eval_runtime".format(data_name): metric.get('eval_runtime')})
  wandb.log({"en-{}-eval_samples_per_second".format(data_name): metric.get('eval_samples_per_second')})
  wandb.log({"en-{}-eval_steps_per_second".format(data_name): metric.get('eval_steps_per_second')})
  
  return


eval_test_lang_2(dataset_en_tok['test'], "en_test" )
eval_test_lang_2(dataset_en_tok['validation'], "en_val" )
