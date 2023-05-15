from transformers import AutoTokenizer, EvalPrediction
from transformers import AutoModelForQuestionAnswering
from tqdm.auto import tqdm
from datasets import load_dataset, load_metric
from transformers import TrainingArguments
from transformers import Trainer
import collections
from transformers import default_data_collator, set_seed
import torch
import numpy as np
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers.adapters import PfeifferConfig, HoulsbyConfig, LoRAConfig, CompacterConfig, PrefixTuningConfig
import argparse
import os
import wandb



parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--reduction_factor", type=int, default=16)
parser.add_argument("--adap_drop", default="noAD", help = "adap drop[AD | noAD]")
parser.add_argument("--run_name", default="its_ok_")
parser.add_argument("--prefix_length", type=int, default=30)
parser.add_argument('--adapter_type', default = "houlsby", help = "adapter type[houlsby|pfeiffer|lora|compacter|houlsbyparallel|pfeifferparallel |prefixtuning]")

args = parser.parse_args()

wandb.init(project="your_project_name", entity="your_entity" , name = f"{args.adapter_type}_{args.adap_drop}_{args.batch_size}" )
set_seed(args.seed)

metric = load_metric("squad")

dataset_squad =  load_dataset("squad")
print("size of dataset squad is : ", dataset_squad)
dataset_squad = dataset_squad.shuffle(seed=42)
print(dataset_squad)

#dataset_squad['train'] = dataset_squad['train'].shard(num_shards=200, index=0)
#dataset_squad['validation'] = dataset_squad['validation'].shard(num_shards=200, index=0)
#print(dataset_squad)
model_name = 'ai4bharat/IndicBERT-MLM-only'
tokenizer = AutoTokenizer.from_pretrained("/nlsasfs/home/ai4bharat/nandinim/nandini/new_adaptertune/qa/tok_InBert_mlm_only/", use_auth_token=True)

max_length = 384
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



model = AutoModelForQuestionAnswering.from_pretrained("/nlsasfs/home/ai4bharat/nandinim/nandini/new_adaptertune/qa/model_InBert_mlm_only_qa_adap/", use_auth_token=True)


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
print(pytorch_total_params)
wandb.log({"total_parameter": pytorch_total_params})

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


train_args = TrainingArguments(
    f"callbacks_inB_qa_{args.adapter_type}_{args.run_name}_{args.adap_drop}_{args.prefix_length}_{args.batch_size}",
    evaluation_strategy= "epoch", #IntervalStrategy.STEPS,# "no",
    save_total_limit = 4,
    save_strategy = "epoch", #IntervalStrategy.STEPS,
    # save_steps =1000 ,
    # logging_steps=1000,
    learning_rate=  args.learning_rate, #1e-5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=50,
    weight_decay=args.weight_decay,
    warmup_steps= 2000,
    fp16=True,
    overwrite_output_dir=True,
    load_best_model_at_end=True,
    metric_for_best_model = 'eval_f1',
    greater_is_better = True,
    seed = args.seed,
)

data_collator = default_data_collator

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset_train,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_2,
    callbacks = [EarlyStoppingCallback(early_stopping_patience= 3 )],
)
#trainer.add_callback(AdapterDropTrainerCallback())
if args.adap_drop == "AD":
  trainer.add_callback(AdapterDropTrainerCallback())

trainer.train()

#model.save_adapter(f"/nlsasfs/home/ai4bharat/nandinim/nandini/new_adaptertune/adapter_fusion/qa_its_ok", "pfeiffer_adapter")


predictions, _, _ = trainer.predict(validation_dataset)
start_logits, end_logits = predictions
print(compute_metrics(start_logits, end_logits, validation_dataset, dataset_squad["validation"]))


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
    print("for dataset ", data_name, " dataset size is : ", dataset)
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
