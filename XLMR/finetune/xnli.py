from datasets import  Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EvalPrediction
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers import BertTokenizer
from transformers import AutoConfig
import numpy as np
import argparse
import numpy as np
import wandb
import os


#os.environ["WANDB_MODE"] = "offline"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--model_name", default="xlmr-b", help= "model type [xlmr-b | xlmr-l| indicbert]")

args = parser.parse_args()

# wandb.init(project="Indicbert_mlm_only_xnli", entity="nandinimundra", name = f"FT_{args.batch_size}_{args.learning_rate}_2")

# model_name = 'ai4bharat/IndicBERT-MLM-only'
# tokenizer = AutoTokenizer.from_pretrained("/nlsasfs/home/ai4bharat/nandinim/nandini/finetune/tok_InBert_mlm_only/", use_auth_token=True)
id2label= {0: "entailment", 1: "neutral", 2: "contradiction"}
label2id = {"entailment":0, "neutral": 1, "contradiction":2 }


if args.model_name == "indicbert":
    wandb.init(project="Indicbert_mlm_only_xnli", entity="nandinimundra", name = f"2_{args.model_name}_FT_{args.batch_size}_{args.learning_rate}")
    model_name = 'ai4bharat/IndicBERT-MLM-only'
    tokenizer = AutoTokenizer.from_pretrained("/nlsasfs/home/ai4bharat/nandinim/nandini/new_adaptertune/xnli/tok_InBert_mlm_only/", use_auth_token=True)
    config = AutoConfig.from_pretrained(
        "/nlsasfs/home/ai4bharat/nandinim/nandini/new_adaptertune/xnli/config_InBert_mlm_only_xnli_adap/",
        num_labels=3,
        id2label={ 0: "entailment", 1: "neutral", 2: "contradiction"},
        use_auth_token=True,
    )
    model = BertForSequenceClassification.from_pretrained(
        "/nlsasfs/home/ai4bharat/nandinim/nandini/new_adaptertune/xnli/model_InBert_mlm_only_xnli_adap/",
        config=config,
        use_auth_token=True,
    )

elif args.model_name == "xlmr-b":
    wandb.init(project="XLMR-base_xnli", entity="nandinimundra", name = f"{args.model_name}_its_ok_{args.batch_size}_{args.learning_rate}")
    #model_name = 'ai4bharat/IndicBERT-MLM-only'
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_auth_token=True)
    config = AutoConfig.from_pretrained(
        "xlm-roberta-base",
        num_labels=3,
        id2label={ 0: "entailment", 1: "neutral", 2: "contradiction"},
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        config=config,
    )
elif args.model_name == "xlmr-l":
    wandb.init(project="xlmr_large_xnli", entity="nandinimundra", name = f"{args.model_name}_its_ok_{args.batch_size}_{args.learning_rate}")
    #model_name = 'ai4bharat/IndicBERT-MLM-only'
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large", use_auth_token=True)
    config = AutoConfig.from_pretrained(
        "xlm-roberta-large",
        num_labels=3,
        id2label={ 0: "entailment", 1: "neutral", 2: "contradiction"},
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-large",
        config=config,
    )

def preprocess(dataset):
  dataset = dataset.map(lambda batch: tokenizer.encode_plus( batch['premise'], batch['hypothesis'], max_length= 128, pad_to_max_length = True, truncation=True))
  dataset = dataset.rename_column("label", "labels")
  dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
  return dataset
  


dataset_en = load_dataset("xnli", 'en')
datset_en = dataset_en.shuffle(seed=5)
#dataset_en['train'] = dataset_en['train'].shard(num_shards=200, index=0)
dataset_ar = load_dataset("xnli", 'ar')
dataset_bg = load_dataset("xnli", 'bg')
dataset_de = load_dataset("xnli", 'de')
dataset_el = load_dataset("xnli", 'el')
dataset_es = load_dataset("xnli", 'es')
dataset_fr = load_dataset("xnli", 'fr')
dataset_hi = load_dataset("xnli", 'hi')
dataset_ru = load_dataset("xnli", 'ru')
dataset_sw = load_dataset("xnli", 'sw')
dataset_th = load_dataset("xnli", 'th')
dataset_tr = load_dataset("xnli", 'tr')
dataset_ur = load_dataset("xnli", 'ur')
dataset_vi = load_dataset("xnli", 'vi')
dataset_zh = load_dataset("xnli", 'zh')


dataset_tok_en = preprocess(dataset_en)
# val_as = preprocess(dataset_as['validation'])
# val_bn = preprocess(dataset_bn['validation'])
# val_hi = preprocess(dataset_hi['validation'])
# val_pa = preprocess(dataset_pa['validation'])
# val_ta = preprocess(dataset_ta['validation'])
# val_te = preprocess(dataset_te['validation'])

test_ar = preprocess(dataset_ar['test'])
test_bg = preprocess(dataset_bg['test'])
test_de = preprocess(dataset_de['test'])
test_el = preprocess(dataset_el['test'])
test_es = preprocess(dataset_es['test'])
test_fr = preprocess(dataset_fr['test'])
test_hi = preprocess(dataset_hi['test'])
test_ru = preprocess(dataset_ru['test'])
test_sw = preprocess(dataset_sw['test'])
test_th = preprocess(dataset_th['test'])
test_tr = preprocess(dataset_tr['test'])
test_ur = preprocess(dataset_ur['test'])
test_vi = preprocess(dataset_vi['test'])
test_zh = preprocess(dataset_zh['test'])



pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
wandb.log({"no_of_parameter": pytorch_total_params})

training_args = TrainingArguments(
    f"callbacks_inB_Senti_FT{args.batch_size}_{args.learning_rate}_{args.model_name}_its_ok",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    warmup_steps= 2000,
    evaluation_strategy = "epoch", #IntervalStrategy.STEPS,
    #eval_steps = 1000,
    save_total_limit = 3,
    num_train_epochs=50,
    save_strategy = "epoch", #IntervalStrategy.STEPS,
    # save_steps =1000 ,
    # logging_steps=1000,
    #output_dir="./training_output",
    metric_for_best_model = 'eval_acc',
    greater_is_better = True,
    overwrite_output_dir=True,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    fp16=True,
)
wandb.log({"batch_size_train":training_args.per_device_train_batch_size })
def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  print("in compute accuracy ******************************************")
  return {"acc": (preds == p.label_ids).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= dataset_tok_en["train"],
    eval_dataset= dataset_tok_en["validation"],
    compute_metrics=compute_accuracy,
    callbacks = [EarlyStoppingCallback(early_stopping_patience= 3 )]
  )

print(trainer.train())
print(trainer.evaluate())

def eval_test_lang(data_test, data_name):
  print("zero shot test performance for lang:   ", data_name )
  eval_trainer = Trainer(
    model=model,
    args= training_args, #TrainingArguments(output_dir="./eval_output", remove_unused_columns=False,),
    eval_dataset= data_test,
    compute_metrics=compute_accuracy,
    )
  metric = eval_trainer.evaluate()
  print("the metric for language ", data_name , " is : ",  metric)
  wandb.log({"en-{}".format(data_name): metric})
  wandb.log({"en-{}-eval_loss".format(data_name): metric.get('eval_loss')})
  wandb.log({"en-{}-eval_acc".format(data_name): metric.get('eval_acc')})
  wandb.log({"en-{}-eval_runtime".format(data_name): metric.get('eval_runtime')})
  wandb.log({"en-{}-eval_steps_per_second".format(data_name): metric.get('eval_steps_per_second')})
  return

eval_test_lang(dataset_tok_en['test'], "en_test" )
eval_test_lang(test_ar, "ar") 
eval_test_lang(test_bg, "bg") 
eval_test_lang(test_de, "de") 
eval_test_lang(test_el, "el") 
eval_test_lang(test_es, "es") 
eval_test_lang(test_fr, "fr") 
eval_test_lang(test_hi, "hi") 
eval_test_lang(test_ru, "ru") 
eval_test_lang(test_sw, "sw") 
eval_test_lang(test_th, "th") 
eval_test_lang(test_tr, "tr") 
eval_test_lang(test_ur, "ur") 
eval_test_lang(test_vi, "vi") 
eval_test_lang(test_zh, "zh") 
eval_test_lang(dataset_tok_en['validation'], "en_val" )



