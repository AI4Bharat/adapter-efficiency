import argparse
import numpy as np
from datasets import load_dataset, load_metric
from torch import nn
from transformers import BertForSequenceClassification, AutoTokenizer,Trainer, TrainingArguments,set_seed
from transformers import EarlyStoppingCallback, IntervalStrategy
import torch, transformers
import os
import wandb


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--model_path", type=str, default="callbacks_MTL_apart__ner_32_3e-05_7")
parser.add_argument("--checkpont_p", type=str, default="checkpoint-69861")
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--max_seq_length", type=int, default=256)
args = parser.parse_args()

wandb.init(project="IndicBert_MLM_only_Sentiment", entity="nandinimundra" , name = f"last_please_{args.model_path}_{args.checkpont_p}" )
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#set_seed(args.seed)

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


metric = load_metric('glue', 'sst2')
dataset_en = preprocess_function(dataset)
#model_name = 'ai4bharat/IndicBERT-MLM-only'

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





def eval_test_lang(data_test, data_name):
  print("zero shot test performance for lang:   ", data_name )
  eval_trainer = Trainer(
    model=multitask_model.taskmodels_dict["sentiment"],
    args=TrainingArguments(output_dir="./eval_output", remove_unused_columns=False,),
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
