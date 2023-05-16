from datasets import  Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoAdapterModel
from transformers import TrainingArguments,  EvalPrediction, AdapterTrainer
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers.adapters import PfeifferConfig, HoulsbyConfig, LoRAConfig, CompacterConfig, PrefixTuningConfig
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
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--reduction_factor", type=int, default=16)
parser.add_argument("--prefix_length", type=int, default=30)
parser.add_argument("--adap_drop", default="noAD", help = "adapter type[AD | noAD]")
parser.add_argument("--run_name", default="its_ok_")
parser.add_argument('--adapter_type', default = "houlsby", help = "adapter type[houlsby|pfeiffer|lora|compacter|houlsbyparallel|pfeifferparallel |prefixtuning]")

args = parser.parse_args()

wandb.init(project="your_project_name", entity="your_entity", name = f"{args.run_name}_{args.adapter_type}_{args.adap_drop}_{args.prefix_length}_{args.batch_size}")

model_name = 'ai4bharat/IndicBERT-MLM-only'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
def preprocess(dataset):
  dataset = dataset.map(lambda batch: tokenizer.encode_plus( batch['premise'], batch['hypothesis'], max_length= 128, pad_to_max_length = True, truncation=True))
  dataset = dataset.rename_column("label", "labels")
  dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
  return dataset
  


dataset_en = load_dataset("xnli", 'en')
print("size of en dataset is : ", dataset_en)
datset_en = dataset_en.shuffle(seed=42)
dataset_as = load_dataset("Divyanshu/indicxnli", 'as')
dataset_bn = load_dataset("Divyanshu/indicxnli", 'bn')
dataset_gu = load_dataset("Divyanshu/indicxnli", 'gu')
dataset_hi = load_dataset("Divyanshu/indicxnli", 'hi')
dataset_kn = load_dataset("Divyanshu/indicxnli", 'kn')
dataset_ml = load_dataset("Divyanshu/indicxnli", 'ml')
dataset_mr = load_dataset("Divyanshu/indicxnli", 'mr')
dataset_or = load_dataset("Divyanshu/indicxnli", 'or')
dataset_pa = load_dataset("Divyanshu/indicxnli", 'pa')
dataset_ta = load_dataset("Divyanshu/indicxnli", 'ta')
dataset_te = load_dataset("Divyanshu/indicxnli", 'te')

print(dataset_as)

dataset_tok_en = preprocess(dataset_en)
val_as = preprocess(dataset_as['validation'])
val_bn = preprocess(dataset_bn['validation'])
val_hi = preprocess(dataset_hi['validation'])
val_pa = preprocess(dataset_pa['validation'])
val_ta = preprocess(dataset_ta['validation'])
val_te = preprocess(dataset_te['validation'])

test_as = preprocess(dataset_as['test'])
test_bn = preprocess(dataset_bn['test'])
test_hi = preprocess(dataset_hi['test'])
test_gu = preprocess(dataset_gu['test'])
test_kn = preprocess(dataset_kn['test'])
test_ml = preprocess(dataset_ml['test'])
test_mr = preprocess(dataset_mr['test'])
test_or = preprocess(dataset_or['test'])
test_pa = preprocess(dataset_pa['test'])
test_ta = preprocess(dataset_ta['test'])
test_te = preprocess(dataset_te['test'])


id2label= {0: "entailment", 1: "neutral", 2: "contradiction"}
label2id = {"entailment":0, "neutral": 1, "contradiction":2 }

config = AutoConfig.from_pretrained(
    model_name,
    num_labels=3,
    id2label={ 0: "entailment", 1: "neutral", 2: "contradiction"},
    use_auth_token=True,
)
model = AutoAdapterModel.from_pretrained(
    model_name,
    config=config,
    use_auth_token=True,
)

if args.adapter_type == "houlsby":
  config_a = HoulsbyConfig(reduction_factor = args.reduction_factor)
  model.add_adapter("houlsby_adapter", config=config_a)
  model.add_classification_head("houlsby_adapter", num_labels=3)
  model.train_adapter("houlsby_adapter")
  adapter_name = "houlsby_adapter"

elif args.adapter_type == "pfeiffer":
  config_a = PfeifferConfig(reduction_factor = args.reduction_factor)
  model.add_adapter("pfeiffer_adapter", config=config_a)
  model.add_classification_head("pfeiffer_adapter", num_labels=3)
  model.train_adapter("pfeiffer_adapter")
  adapter_name = "pfeiffer_adapter"

elif args.adapter_type == "lora":
  config_a = LoRAConfig(r=8, alpha=16)
  model.add_adapter("lora_adapter", config=config_a)
  model.add_classification_head("lora_adapter", num_labels=3)
  model.train_adapter("lora_adapter")
  adapter_name = "lora_adapter"

elif args.adapter_type == "compacter":
  config_a = CompacterConfig()
  model.add_adapter("Compacter_adapter", config=config_a)
  model.add_classification_head("Compacter_adapter", num_labels=3)
  model.train_adapter("Compacter_adapter")
  adapter_name = "Compacter_adapter"

elif args.adapter_type == "houlsbyparallel":
  config_a = HoulsbyConfig(is_parallel = True)
  model.add_adapter("houlsby_P_adapter", config=config_a)
  model.add_classification_head("houlsby_P_adapter", num_labels=3)
  model.train_adapter("houlsby_P_adapter")
  adapter_name = "houlsby_P_adapter"

elif args.adapter_type == "pfeifferparallel":
  config_a = PfeifferConfig(is_parallel = True)
  model.add_adapter("pfeiffer_P_adapter", config=config_a)
  model.add_classification_head("pfeiffer_P_adapter", num_labels=3)
  model.train_adapter("pfeiffer_P_adapter")
  adapter_name = "pfeiffer_P_adapter"

elif args.adapter_type == "prefixtuning":
  config_a = PrefixTuningConfig(flat=False, prefix_length=args.prefix_length)
  model.add_adapter("prefix_adapter", config=config_a)
  model.add_classification_head("prefix_adapter", num_labels=3)
  model.train_adapter("prefix_adapter")
  adapter_name = "prefix_adapter"



# for name, param in model.named_parameters():
#      print(name, param.requires_grad)
# print(model)

# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
wandb.log({"no_of_parameter": pytorch_total_params})



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
    f"callbacks_inB_xnli_{args.adapter_type}_{args.run_name}_{args.adap_drop}_{args.prefix_length}_{args.batch_size}_2",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    warmup_steps= 2000,
    evaluation_strategy = "epoch", #IntervalStrategy.STEPS,
    #eval_steps = 1000,
    save_total_limit = 5,
    num_train_epochs=50,
    save_strategy = "epoch", #IntervalStrategy.STEPS,
    # save_steps = 1000,
    # logging_steps= 1000,
    #output_dir="./training_output",
    metric_for_best_model = 'eval_acc',
    greater_is_better = True,
    overwrite_output_dir=True,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    fp16=True,
)
#wandb.log({"batch_size_train":training_args.per_device_train_batch_size })
def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  print("in compute accuracy ******************************************")
  return {"acc": (preds == p.label_ids).mean()}

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset= dataset_tok_en["train"],
    eval_dataset= dataset_tok_en["validation"],
    compute_metrics=compute_accuracy,
    callbacks = [EarlyStoppingCallback(early_stopping_patience= 3 )]
  )
if args.adap_drop == "AD":
  trainer.add_callback(AdapterDropTrainerCallback())

print(trainer.train())
print(trainer.evaluate())
#model.save_adapter(f"/xnli_its_ok", "pfeiffer_adapter")


def eval_test_lang(data_test, data_name):
  print("for dataset ", data_name, " dataset size is : ", data_test)
  print("zero shot test performance for lang:   ", data_name )
  eval_trainer = AdapterTrainer(
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
eval_test_lang(test_as, "as" )
eval_test_lang(test_bn, "bn" )
eval_test_lang(test_hi, "hi" )
eval_test_lang(test_gu, "gu" )
eval_test_lang(test_kn, "kn" )
eval_test_lang(test_ml, "ml" )
eval_test_lang(test_mr, "mr" )
eval_test_lang(test_or, "or" )
eval_test_lang(test_pa, "pa" )
eval_test_lang(test_ta, "ta" )
eval_test_lang(test_te, "te" )
eval_test_lang(dataset_tok_en['validation'], "en_val" )
