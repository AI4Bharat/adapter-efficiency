from datasets import  Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import BertForSequenceClassification
from transformers import TrainingArguments, Trainer, EvalPrediction
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers import BertTokenizer
from transformers import AutoConfig
import numpy as np
from torch import nn
import argparse, torch
import numpy as np
import wandb
import os, transformers


#os.environ["WANDB_MODE"] = "offline"
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model_path", type=str, default="callbacks_MTL_apart__ner_32_3e-05_7")
parser.add_argument("--checkpont_p", type=str, default="checkpoint-69861")
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--warmup_ratio", type=float, default=0.1)

args = parser.parse_args()

wandb.init(project="Indicbert_mlm_only_xnli", entity="your_entity", name = f"last_please_{args.model_path}_{args.checkpont_p}")

model_name = 'ai4bharat/IndicBERT-MLM-only'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
def add_prefix(example):
    example["premise"] = 'premise: ' + example["premise"]
    example["hypothesis"] = 'hypothesis: ' + example["hypothesis"]
    return example
def preprocess(dataset):
    dataset = dataset.map(add_prefix)
    dataset = dataset.map(lambda batch: tokenizer.encode_plus( batch['premise'], batch['hypothesis'], max_length= 256, pad_to_max_length = True, truncation=True))
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset
  
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
dataset_en = load_dataset("xnli", 'en')
datset_en = dataset_en.shuffle(seed=42)
#dataset_en['train'] = dataset_en['train'].shard(num_shards=200, index=0)
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



def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  print("in compute accuracy ******************************************")
  return {"acc": (preds == p.label_ids).mean()}

def eval_test_lang(data_test, data_name):
  print("zero shot test performance for lang:   ", data_name )
  eval_trainer = Trainer(
    model=multitask_model.taskmodels_dict["xnli"],
    args= TrainingArguments(output_dir="./eval_output", remove_unused_columns=False,),
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

