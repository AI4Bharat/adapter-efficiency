from datasets import  Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoAdapterModel
from transformers import TrainingArguments,  EvalPrediction, AdapterTrainer
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction, TrainerCallback
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers.adapters import PfeifferConfig, HoulsbyConfig, LoRAConfig, CompacterConfig, PrefixTuningConfig
from transformers import BertTokenizer
from transformers import AutoConfig
from transformers.adapters.composition import Fuse
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
parser.add_argument('--adapter_type', default = "pfeiffer", help = "adapter type[houlsby|pfeiffer|lora|compacter|houlsbyparallel|pfeifferparallel |prefixtuning]")
parser.add_argument('--apart', default = "none", help = "adapter type[ner|sentiment|paraphrase|copa|qa]")

args = parser.parse_args()

wandb.init(project="Indicbert_mlm_only_xnli", entity="", nyour_entityame = f"adapter_fusion_fused_its_ok_{args.adapter_type}_{args.batch_size}_2")

model_name = 'ai4bharat/IndicBERT-MLM-only'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
def preprocess(dataset):
  dataset = dataset.map(lambda batch: tokenizer.encode_plus( batch['premise'], batch['hypothesis'], max_length= 128, pad_to_max_length = True, truncation=True))
  dataset = dataset.rename_column("label", "labels")
  dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
  return dataset
  


dataset_en = load_dataset("xnli", 'en')
dataset_en['train'] = dataset_en['train'].shard(num_shards=200, index=0)
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


dataset_tok_en = preprocess(dataset_en)
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


if args.apart == "ner":
    print("in ner")
    model.load_adapter("copa_32", load_as="copa", with_head=False)
    #model.load_adapter("ner_32", load_as="ner", with_head=False)
    model.load_adapter("xnli_32", load_as="xnli", with_head=False)
    model.load_adapter("qa_32", load_as="qa", with_head=False)
    model.load_adapter("sentiment_32", load_as="sentiment", with_head=False)
    model.load_adapter("paraphrase_32", load_as="paraphrase", with_head=False)
    model.add_adapter_fusion(Fuse("copa", "xnli",   "paraphrase", "qa", "sentiment"))
    model.set_active_adapters(Fuse("copa", "xnli",  "paraphrase",  "qa", "sentiment"))
    model.add_classification_head("class_head", num_labels=3)
    adapter_setup = Fuse("copa", "xnli",  "paraphrase", "qa", "sentiment")
    model.train_adapter_fusion(adapter_setup)

elif args.apart == "copa":
    print("in copa")
    #model.load_adapter("copa_32", load_as="copa", with_head=False)
    model.load_adapter("ner_32", load_as="ner", with_head=False)
    model.load_adapter("xnli_32", load_as="xnli", with_head=False)
    model.load_adapter("qa_32", load_as="qa", with_head=False)
    model.load_adapter("sentiment_32", load_as="sentiment", with_head=False)
    model.load_adapter("paraphrase_32", load_as="paraphrase", with_head=False)

    # Add a fusion layer for all loaded adapters
    model.add_adapter_fusion(Fuse( "ner", "xnli",   "paraphrase", "qa", "sentiment"))
    model.set_active_adapters(Fuse( "ner", "xnli",  "paraphrase",  "qa", "sentiment"))
    model.add_classification_head("class_head", num_labels=3)
    adapter_setup = Fuse( "ner", "xnli",  "paraphrase", "qa", "sentiment")
    model.train_adapter_fusion(adapter_setup)

elif args.apart == "paraphrase":
    print("in paraphrase")
    model.load_adapter("copa_32", load_as="copa", with_head=False)
    model.load_adapter("ner_32", load_as="ner", with_head=False)
    model.load_adapter("xnli_32", load_as="xnli", with_head=False)
    model.load_adapter("qa_32", load_as="qa", with_head=False)
    model.load_adapter("sentiment_32", load_as="sentiment", with_head=False)
    #model.load_adapter("paraphrase_32", load_as="paraphrase", with_head=False)

    # Add a fusion layer for all loaded adapters
    model.add_adapter_fusion(Fuse("copa", "ner", "xnli",  "qa", "sentiment"))
    model.set_active_adapters(Fuse("copa", "ner", "xnli",  "qa", "sentiment"))
    model.add_classification_head("class_head", num_labels=3)
    adapter_setup = Fuse("copa", "ner", "xnli", "qa", "sentiment")
    model.train_adapter_fusion(adapter_setup)

elif args.apart == "sentiment":
    print("in sentiment")
    model.load_adapter("copa_32", load_as="copa", with_head=False)
    model.load_adapter("ner_32", load_as="ner", with_head=False)
    model.load_adapter("xnli_32", load_as="xnli", with_head=False)
    model.load_adapter("qa_32", load_as="qa", with_head=False)
    #model.load_adapter("sentiment_32", load_as="sentiment", with_head=False)
    model.load_adapter("paraphrase_32", load_as="paraphrase", with_head=False)

    # Add a fusion layer for all loaded adapters
    model.add_adapter_fusion(Fuse("copa", "ner", "xnli",   "paraphrase", "qa"))
    model.set_active_adapters(Fuse("copa", "ner", "xnli",  "paraphrase",  "qa"))
    model.add_classification_head("class_head", num_labels=3)
    adapter_setup = Fuse("copa", "ner", "xnli",  "paraphrase", "qa")
    model.train_adapter_fusion(adapter_setup)

elif args.apart == "qa":
    print("in qa")
    model.load_adapter("copa_32", load_as="copa", with_head=False)
    model.load_adapter("ner_32", load_as="ner", with_head=False)
    model.load_adapter("xnli_32", load_as="xnli", with_head=False)
    #model.load_adapter("qa_32", load_as="qa", with_head=False)
    model.load_adapter("sentiment_32", load_as="sentiment", with_head=False)
    model.load_adapter("paraphrase_32", load_as="paraphrase", with_head=False)

    # Add a fusion layer for all loaded adapters
    model.add_adapter_fusion(Fuse("copa", "ner", "xnli",   "paraphrase", "sentiment"))
    model.set_active_adapters(Fuse("copa", "ner", "xnli",  "paraphrase", "sentiment"))
    model.add_classification_head("class_head", num_labels=3)
    
    adapter_setup = Fuse("copa", "ner", "xnli",  "paraphrase", "sentiment")
    model.train_adapter_fusion(adapter_setup)

else :
    print("in none")
    model.load_adapter("copa_its_ok", load_as="copa", with_head=False)
    model.load_adapter("ner_its_ok", load_as="ner", with_head=False)
    model.load_adapter("xnli_its_ok", load_as="xnli", with_head=False)
    model.load_adapter("qa_its_ok", load_as="qa", with_head=False)
    model.load_adapter("sentiment_its_ok", load_as="sentiment", with_head=False)
    model.load_adapter("paraphrase_its_ok", load_as="paraphrase", with_head=False)

    # Add a fusion layer for all loaded adapters
    model.add_adapter_fusion(Fuse("copa", "ner", "xnli",   "paraphrase", "qa", "sentiment"))
    model.set_active_adapters(Fuse("copa", "ner", "xnli",  "paraphrase",  "qa", "sentiment"))
    model.add_classification_head("class_head", num_labels=3)
    adapter_setup = Fuse("copa", "ner", "xnli",  "paraphrase", "qa", "sentiment")
    model.train_adapter_fusion(adapter_setup)

# model.load_adapter("copa_32", load_as="copa", with_head=False)
# model.load_adapter("ner_32", load_as="ner", with_head=False)
# model.load_adapter("xnli_32", load_as="xnli", with_head=False)
# model.load_adapter("qa_32", load_as="qa", with_head=False)
# model.load_adapter("sentiment_32", load_as="sentiment", with_head=False)
# model.load_adapter("paraphrase_32", load_as="paraphrase", with_head=False)

# Add a fusion layer for all loaded adapters
# model.add_adapter_fusion(Fuse("copa", "ner", "xnli",   "paraphrase", "qa", "sentiment"))
# model.set_active_adapters(Fuse("copa", "ner", "xnli",  "paraphrase", "qa", "sentiment"))
#model.add_classification_head("class_head", num_labels=3)



# adapter_setup = Fuse("copa", "ner", "xnli",  "paraphrase", "qa", "sentiment")
# model.train_adapter_fusion(adapter_setup)
for name, param in model.named_parameters():
     print(name, param.requires_grad)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
wandb.log({"no_of_parameter": pytorch_total_params})


training_args = TrainingArguments(
    f"callbacks_inB_xnli_apart_{args.apart}_{args.batch_size}_its_ok",
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
    # #output_dir="./training_output",
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

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset= dataset_tok_en["train"],
    eval_dataset= dataset_tok_en["validation"],
    compute_metrics=compute_accuracy,
    callbacks = [EarlyStoppingCallback(early_stopping_patience= 3 )]
  )

print(trainer.train())
print(trainer.evaluate())
#model.save_adapter_fusion("xnli_f/fuse", "copa,ner,xnli,paraphrase,qa,sentiment")
#model.save_all_adapters("xnli_f/all")

def eval_test_lang(data_test, data_name):
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
