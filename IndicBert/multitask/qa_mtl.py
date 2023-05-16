from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, EvalPrediction
from transformers import AutoModelForQuestionAnswering
from tqdm.auto import tqdm
from transformers import TrainingArguments
from transformers import Trainer
import collections
from transformers import default_data_collator
import torch, transformers
from torch import nn
import numpy as np
from transformers import EarlyStoppingCallback, IntervalStrategy
import argparse
import os
import wandb



parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--model_path", type=str, default="callbacks_MTL_xnli_pref_ep_10_8__32_4e-05_7")
parser.add_argument("--checkpont_p", type=str, default="checkpoint-69861")

args = parser.parse_args()

wandb.init(project="IndicBert_mlm_only_qa",  entity="your_entity" , name = f"last_please_{args.model_path}_{args.checkpont_p}")


metric = load_metric("squad")

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


dataset_squad =  load_dataset("squad")
dataset_squad = dataset_squad.shuffle(seed=42)
print(dataset_squad)

#dataset_squad['train'] = dataset_squad['train'].shard(num_shards=100, index=0)
#dataset_squad['validation'] = dataset_squad['validation'].shard(num_shards=100, index=0)
#print(dataset_squad)
model_name = 'ai4bharat/IndicBERT-MLM-only'
max_length = 256#384
stride = 128
n_best_size = 20
squad_v2 = False
max_answer_length = 30
predicted_answers = []
n_best = 20


def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


train_dataset = dataset_squad["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=dataset_squad["train"].column_names,
)

def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


validation_dataset = dataset_squad["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=dataset_squad["validation"].column_names,
)
validation_dataset_train = dataset_squad["validation"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=dataset_squad["validation"].column_names,
)
print(len(dataset_squad["validation"]), len(validation_dataset))
print("validation dataset after tokenizing ", validation_dataset)
#print(validation_dataset[0])

def compute_metrics(start_logits, end_logits, features, examples):
    print("bla bla in compute metric")
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)



def compute_metrics_2(p: EvalPrediction):

    print("bla bla in compute metric 2")

    predictions, _, _ = trainer.predict(validation_dataset)
    start_logits, end_logits = predictions
    
    #start_logits, end_logits = p.predictions
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(validation_dataset):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(dataset_squad["validation"]):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = validation_dataset[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset_squad["validation"]]
    m = metric.compute(predictions=predicted_answers, references=theoretical_answers)
    # add 'eval_' to the metric name to solve bug with trainer
    return {f"eval_{k}" if "eval_" not in k else k: v for k, v in m.items()}
    #return metric.compute(predictions=predicted_answers, references=theoretical_answers)


args = TrainingArguments(
    f"callbacks_inB_Senti_FT{args.batch_size}_{args.learning_rate}",
    evaluation_strategy= IntervalStrategy.STEPS,# "no",
    save_total_limit = 5,
    save_strategy = IntervalStrategy.STEPS,
    save_steps =1000 ,
    logging_steps=1000,
    learning_rate=  args.learning_rate, #1e-5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=50,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    fp16=True,
    overwrite_output_dir=True,
    load_best_model_at_end=True,
    metric_for_best_model = 'eval_f1',
    greater_is_better = True,
    seed = args.seed,
)

data_collator = default_data_collator

trainer = Trainer(
    model=multitask_model.taskmodels_dict["qa"],
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset_train,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_2,
    callbacks = [EarlyStoppingCallback(early_stopping_patience= 5 )],
)
#trainer.train()

predictions, _, _ = trainer.predict(validation_dataset)
start_logits, end_logits = predictions
print(compute_metrics(start_logits, end_logits, validation_dataset, dataset_squad["validation"]))

dataset_squad =  load_dataset("squad")


dataset_en = load_dataset("xquad", "xquad.en")

dataset_as = load_dataset("ai4bharat/IndicQA","indicqa.as", use_auth_token=True)
dataset_bn = load_dataset("ai4bharat/IndicQA","indicqa.bn", use_auth_token=True)
dataset_gu = load_dataset("ai4bharat/IndicQA","indicqa.gu", use_auth_token=True)
dataset_hi = load_dataset("ai4bharat/IndicQA","indicqa.hi", use_auth_token=True)
dataset_kn = load_dataset("ai4bharat/IndicQA","indicqa.kn", use_auth_token=True)
dataset_ml = load_dataset("ai4bharat/IndicQA","indicqa.ml", use_auth_token=True)
dataset_mr = load_dataset("ai4bharat/IndicQA","indicqa.mr", use_auth_token=True)
dataset_or = load_dataset("ai4bharat/IndicQA","indicqa.or", use_auth_token=True)
dataset_pa = load_dataset("ai4bharat/IndicQA","indicqa.pa", use_auth_token=True)
dataset_ta = load_dataset("ai4bharat/IndicQA","indicqa.ta", use_auth_token=True)
dataset_te = load_dataset("ai4bharat/IndicQA","indicqa.te", use_auth_token=True)

def zero_val(dataset, data_name):
    val_dataset = dataset["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=dataset["validation"].column_names,

    )
    predictions_d, _, _ = trainer.predict(val_dataset)
    start_logits_d, end_logits_d = predictions_d
    metric = compute_metrics(start_logits_d, end_logits_d, val_dataset, dataset["validation"])
    print("the metric for language ", data_name , " is : ",  metric)
    wandb.log({"en-{}".format(data_name): metric})
    wandb.log({"en-{}-exact_match".format(data_name): metric.get('exact_match')})
    wandb.log({"en-{}-eval_f1".format(data_name): metric.get('f1')})
    return
    

zero_val(dataset_squad, 'en_squad')
zero_val(dataset_en, 'en')
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
