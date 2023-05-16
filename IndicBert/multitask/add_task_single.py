''''changes made to and bert for token classification in bert_for_sequence_classification'''
from datasets import load_dataset, concatenate_datasets
from transformers import  DataCollatorForTokenClassification, Trainer, BertForSequenceClassification
import numpy as np
from torch import nn
import transformers, torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
import wandb, os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_path", type=str, default="none")
parser.add_argument("--checkpont_p", type=str, default="none")
parser.add_argument("--shard_num", type=int, default= 10)
#parser.add_argument('--apart', default = "none", help = "apart[xnli|ner|sentiment|paraphrase|copa|qa |none]")
parser.add_argument('--add_task', default = "none", help = "addtask[xnli|ner|sentiment|paraphrase|copa|qa |none]")
parser.add_argument("--model_name", type=str, default="one_only")  # ner was commented
args = parser.parse_args()
wandb.init(project="multitask", entity="your_entity" , name = f"{args.model_name}_{args.add_task}_{args.learning_rate}" )
########################################### XNLI DATA processing
model_name = 'ai4bharat/IndicBERT-MLM-only'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

dataset_xnli = load_dataset("xnli", 'en')
dataset_xnli = dataset_xnli.shuffle(seed=args.seed)

def add_prefix_xnli(example):
    example["premise"] = 'premise: ' + example["premise"]
    example["hypothesis"] = 'hypothesis: ' + example["hypothesis"]
    return example
dataset_xnli = dataset_xnli.map(add_prefix_xnli)
print("after ", dataset_xnli['train'][:5])

def preprocess_xnli(dataset):
  dataset = dataset.map(lambda batch: tokenizer.encode_plus( batch['premise'], batch['hypothesis'], max_length= 128, pad_to_max_length = True, truncation=True))
  return dataset
dataset_xnli = preprocess_xnli(dataset_xnli)

print("after tokenizing: ", dataset_xnli)
#dataset_xnli['train'] = dataset_xnli['train'].shard(num_shards=2, index=0)

###########################################NER data processing ##################################

dataset_ner =  load_dataset("conll2003")
#dataset_ner['train'] = dataset_ner['train'].shard(num_shards=200, index=0)
dataset_ner = dataset_ner.filter(lambda example: example["ner_tags"].count(7) == 0)
dataset_ner = dataset_ner.filter(lambda example: example["ner_tags"].count(8) == 0)
dataset_ner = dataset_ner.shuffle(seed=args.seed)
labels_ner = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
id_2_label_ner = {id_: label for id_, label in enumerate(labels_ner)}
label_2_id_ner = {label: id_ for id_, label in enumerate(labels_ner)}


##############################################3 paraphrase ##############################33
dataset_paraphrase = load_dataset("paws-x", "en")
dataset_paraphrase = dataset_paraphrase.shuffle(seed=args.seed)

label_list_paraphrase = dataset_paraphrase["train"].features["label"].names
#dataset_paraphrase['train'] = dataset_paraphrase['train'].shard(num_shards=200, index=0)


############################################### SENTIMENT ######################################3
dataset_sentiment = load_dataset("amazon_reviews_multi", "en")
#dataset_sentiment['train'] = dataset_sentiment['train'].shard(num_shards=200, index=0)
dataset_sentiment = dataset_sentiment.filter(lambda example: example["stars"] != 3)
def amazon_mapper_sentiment(example):
    if example["stars"] > 3:
        example["label"] = 1
    elif example["stars"] < 3:
        example["label"] = 0
    return {"sentence1": example["review_body"], "label": example["label"]}

dataset_sentiment = dataset_sentiment.map(amazon_mapper_sentiment, remove_columns=dataset_sentiment['train'].column_names)
dataset_sentiment_validation = dataset_sentiment['validation']
label_list_sentiment = dataset_sentiment['train'].unique('label')

#################################################### qa Dataset ##########################################################
dataset_qa =  load_dataset("squad")
dataset_qa = dataset_qa.shuffle(seed=args.seed)
#dataset_qa['train'] = dataset_qa['train'].shard(num_shards=200, index=0)


################################################# copa Dataset ########################################################3
dataset_copa = load_dataset("social_i_qa")
dataset_copa = dataset_copa.shuffle(seed=args.seed)
#dataset_copa['train'] = dataset_copa['train'].shard(num_shards=200, index=0)
#dataset_copa['validation'] = dataset_copa['validation'].shard(num_shards=20, index=0)
dataset_copa = dataset_copa.rename_column("label", "labels")
def convert_label_to_int_copa(example):
    # for siqa
    example["labels"] = int(example["labels"]) - 1
    return example

dataset_copa = dataset_copa.map(convert_label_to_int_copa)

'''
dataset_copa['train'] = concatenate_datasets([
    dataset_copa['train'], dataset_copa['train'], dataset_copa['train']
])
print(dataset_copa)

dataset_paraphrase['train'] = concatenate_datasets([
    dataset_paraphrase['train'], dataset_paraphrase['train']
])
print(dataset_paraphrase)

dataset_ner['train'] = concatenate_datasets([
    dataset_ner['train'], dataset_ner['train'], dataset_ner['train'], dataset_ner['train'], 
    dataset_ner['train'], dataset_ner['train'], dataset_ner['train'], dataset_ner['train'], dataset_ner['train'], dataset_ner['train']
])
print(dataset_ner)

dataset_qa['train'] = concatenate_datasets([
    dataset_qa['train'], dataset_qa['train']
])
print(dataset_qa)
'''
dataset_qa = dataset_qa.shuffle(seed=args.seed)
dataset_new = dataset_ner.shuffle(seed=args.seed)
dataset_paraphrase = dataset_paraphrase.shuffle(seed=args.seed)
dataset_copa = dataset_copa.shuffle(seed=args.seed)


if args.add_task == "xnli":
    # dataset_ner['train'] = dataset_ner['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_paraphrase['train'] = dataset_paraphrase['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_sentiment['train'] = dataset_sentiment['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_qa['train'] = dataset_qa['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_copa['train'] = dataset_copa['train'].shard(num_shards=args.shard_num, index=0)
    dataset_dict = {
        "xnli": dataset_xnli, #load_dataset("xnli", 'en'),
        #"ner": dataset_ner, #load_dataset("conll2003"),
        #"paraphrase" : dataset_paraphrase,
        #"sentiment": dataset_sentiment,
        #"qa" : dataset_qa,
        #copa": dataset_copa,
    }
    print("in xnli ",dataset_dict)
elif args.add_task == "ner":
    # dataset_xnli['train'] = dataset_xnli['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_paraphrase['train'] = dataset_paraphrase['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_sentiment['train'] = dataset_sentiment['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_qa['train'] = dataset_qa['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_copa['train'] = dataset_copa['train'].shard(num_shards=args.shard_num, index=0)
    dataset_dict = {
        #"xnli": dataset_xnli, 
        "ner": dataset_ner,
        # "paraphrase" : dataset_paraphrase,
        # "sentiment": dataset_sentiment,
        # "qa" : dataset_qa,
        # "copa": dataset_copa,
    }
    print("in ner ",dataset_dict)
elif args.add_task == "paraphrase":
    # dataset_xnli['train'] = dataset_xnli['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_ner['train'] = dataset_ner['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_sentiment['train'] = dataset_sentiment['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_qa['train'] = dataset_qa['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_copa['train'] = dataset_copa['train'].shard(num_shards=args.shard_num, index=0)
    dataset_dict = {
        # "xnli": dataset_xnli, 
        # "ner": dataset_ner,
        "paraphrase" : dataset_paraphrase,
        # "sentiment": dataset_sentiment,
        # "qa" : dataset_qa,
        # "copa": dataset_copa,
    }
    print("in paraphrase ",dataset_dict)
elif args.add_task == "sentiment":
    # dataset_xnli['train'] = dataset_xnli['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_ner['train'] = dataset_ner['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_paraphrase['train'] = dataset_paraphrase['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_qa['train'] = dataset_qa['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_copa['train'] = dataset_copa['train'].shard(num_shards=args.shard_num, index=0)
    dataset_dict = {
        # "xnli": dataset_xnli, 
        # "ner": dataset_ner,
        # "paraphrase" : dataset_paraphrase,
        "sentiment": dataset_sentiment,
        # "qa" : dataset_qa,
        # "copa": dataset_copa,
    }
    print("in sentiment ",dataset_dict)
elif args.add_task == "qa":
    # dataset_xnli['train'] = dataset_xnli['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_ner['train'] = dataset_ner['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_paraphrase['train'] = dataset_paraphrase['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_sentiment['train'] = dataset_sentiment['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_copa['train'] = dataset_copa['train'].shard(num_shards=args.shard_num, index=0)
    dataset_dict = {
        # "xnli": dataset_xnli, 
        # "ner": dataset_ner,
        # "paraphrase" : dataset_paraphrase,
        # "sentiment": dataset_sentiment,
        "qa" : dataset_qa,
        # "copa": dataset_copa,
    }
    print("in qa ",dataset_dict)
elif args.add_task == "copa":
    # dataset_xnli['train'] = dataset_xnli['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_ner['train'] = dataset_ner['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_paraphrase['train'] = dataset_paraphrase['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_sentiment['train'] = dataset_sentiment['train'].shard(num_shards=args.shard_num, index=0)
    # dataset_qa['train'] = dataset_qa['train'].shard(num_shards=args.shard_num, index=0)
    
    dataset_dict = {
        # "xnli": dataset_xnli, 
        # "ner": dataset_ner,
        # "paraphrase" : dataset_paraphrase,
        # "sentiment": dataset_sentiment,
        # "qa" : dataset_qa,
        "copa": dataset_copa,
    }
    print("in copa", dataset_dict)






for task_name, dataset in dataset_dict.items():
    print(task_name)
    print(dataset_dict[task_name])
    print()


#num_labels_xnli = len(dataset_dict['xnli']["train"].features["label"].names)
#print(num_labels_xnli)

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


data_collator = DataCollatorForTokenClassification(tokenizer,)

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
        "ner": transformers.AutoConfig.from_pretrained(model_name, num_labels=len(labels_ner), label2id=label_2_id_ner, id2label=id_2_label_ner, use_auth_token=True),
        "paraphrase": transformers.AutoConfig.from_pretrained(model_name, num_labels=len(label_list_paraphrase), use_auth_token=True),
        "sentiment": transformers.AutoConfig.from_pretrained(model_name, num_labels=len(label_list_sentiment), use_auth_token=True), 
        "qa": transformers.AutoConfig.from_pretrained(model_name, use_auth_token= True ),
        "copa" : transformers.AutoConfig.from_pretrained(model_name, use_auth_token= True ),
    },
)
multitask_model.load_state_dict(torch.load(f"/new_finetune/MTL/{args.model_path}/{args.checkpont_p}/pytorch_model.bin" ))
print("model_weight_loaded")
if model_name.startswith("ai4bharat"):
    #print(multitask_model.encoder.embeddings.word_embeddings.weight.data_ptr())
    #print(multitask_model.taskmodels_dict["xnli"].bert.encoder.layer[0].attention.self.query.weight.data_ptr())
    #print(multitask_model.taskmodels_dict["ner"].bert.embeddings.word_embeddings.weight.data_ptr())
    print(multitask_model.encoder.encoder.layer[0].attention.self.query.weight.data_ptr())
    print(multitask_model.taskmodels_dict["xnli"].bert.encoder.layer[0].attention.self.query.weight.data_ptr())
    print(multitask_model.taskmodels_dict["ner"].bert.encoder.layer[0].attention.self.query.weight.data_ptr())
    print(multitask_model.taskmodels_dict["paraphrase"].bert.encoder.layer[0].attention.self.query.weight.data_ptr())
    print(multitask_model.taskmodels_dict["sentiment"].bert.encoder.layer[0].attention.self.query.weight.data_ptr())
    print(multitask_model.taskmodels_dict["qa"].bert.encoder.layer[0].attention.self.query.weight.data_ptr())
    print(multitask_model.taskmodels_dict["copa"].bert.encoder.layer[0].attention.self.query.weight.data_ptr())
else:
    print("Exercise for the reader: add a check for other model architectures =)")


############################################## TOKENIZATION OF DATASET ###################################################
max_seq_len = 256

def convert_to_xnli_features(example_batch):
    '''inputs = list(zip(example_batch['premise'], example_batch['hypothesis']))
    
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_seq_len, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    features["labels"] = [
            [l] + [-100] * (max_seq_len - 1) for l in features["labels"]
        ]
    return features'''
    #  when pretokenizing xnli
    example_batch["labels"] = example_batch["label"]
    example_batch["labels"] = [
            [l] + [-100] * (max_seq_len - 1) for l in example_batch["labels"]
        ]
    return example_batch

def convert_to_ner_features(examples):
    text_column_name = "tokens"
    label_column_name = "ner_tags"
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        max_length=max_seq_len,
        #padding=False,
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

def convert_to_paraphrase_features(examples):
    inputs = list(zip(examples['sentence1'], examples['sentence2']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_seq_len, pad_to_max_length=True
    )
    features["labels"] = examples["label"]
    features["labels"] = [
            [l] + [-100] * (max_seq_len - 1) for l in features["labels"]
        ]
    return features


def convert_to_sentiment_features(examples):
    inputs = list(examples['sentence1'])
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_seq_len, pad_to_max_length=True
    )
    features["labels"] = examples["label"]
    features["labels"] = [
            [l] + [-100] * (max_seq_len - 1) for l in features["labels"]
        ]
    return features
    


#################################################### Tokenization of SQUAD #############################################33
stride = 128
n_best_size = 20
squad_v2 = False
max_answer_length = 30
predicted_answers = []
n_best = 20
def convert_to_qa_features(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_seq_len,
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
    #inputs["labels"] = [
     #       [s] + [e] + [-100] * (max_seq_len - 1) for s, e in zip(inputs["start_positions"],inputs["end_positions"] )
      #  ]
    return inputs

################################################# convert of copa dataset ##############################

def convert_to_copa_features(examples):

    ending_names = [f"answer{i}" for i in "ABC"]
    context_name = "context"
    question_header_name = "question"

    first_sentences = [
        [context] * len(ending_names) for context in examples[context_name]
    ]
    question_headers = examples[question_header_name]
    
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names]
        for i, header in enumerate(question_headers)
    ]

    # Flatten out
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length= max_seq_len,
        pad_to_max_length = True,
        # padding="max_length",
    )
    # Un-flatten
    return {
        k: [
            v[i : i + len(ending_names)]
            for i in range(0, len(v), len(ending_names))
        ]
        for k, v in tokenized_examples.items()
    }

def label_convert_copa(examples):
    examples["labels"] = [
            [l] + [-100] * (256 - 1) for l in examples["labels"]
        ]
    return examples




convert_func_dict = {
    "xnli": convert_to_xnli_features,
    "ner": convert_to_ner_features,
    "paraphrase": convert_to_paraphrase_features,
    "sentiment": convert_to_sentiment_features,
    "qa": convert_to_qa_features,
    "copa": convert_to_copa_features,
}

columns_dict = {
    "xnli": ['input_ids', 'attention_mask','labels',  'token_type_ids'],
    "ner": ['input_ids', 'attention_mask', 'labels', 'token_type_ids'],
    "paraphrase": ['input_ids', 'attention_mask', 'labels', 'token_type_ids'],
    "sentiment": ['input_ids', 'attention_mask', 'labels', 'token_type_ids'],
    "qa": ['input_ids', 'attention_mask',  'token_type_ids'],     #needs to be changed
    "copa":['input_ids', 'attention_mask', 'labels', 'token_type_ids'],
    
}
columns_remove = {
    "xnli": [ "premise", "hypothesis", "label"],
    "ner": ["chunk_tags", "id", "ner_tags", "pos_tags", "tokens"],
    "paraphrase": [ "sentence1", "sentence2", "label", "id"],
    "sentiment": [ "sentence1", "label"],
    "qa": ['id', 'title', 'context', 'question', 'answers'],
    "copa": ['context','question', 'answerA', 'answerC', 'answerB' ],
}

features_dict = {}

for task_name, dataset in dataset_dict.items():
    print("task name is ", task_name)
    print("dataset is ", dataset)
    features_dict[task_name] = {}
    for phase, phase_dataset in dataset.items():
        print("phase is ", phase)
        print("phase dataset is ", phase_dataset)
        features_dict[task_name][phase] = phase_dataset.map(
            convert_func_dict[task_name],
            batched=True,
            load_from_cache_file=False,
            remove_columns = columns_remove[task_name],
        )
        if task_name == "copa":
            features_dict[task_name][phase] = features_dict[task_name][phase].map(
                label_convert_copa, 
                batched=True,
            )

        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
        if task_name == "xnli":
            features_dict[task_name][phase].set_format(
                type="torch", 
                columns=columns_dict[task_name],
            )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))


print("features dict is ", features_dict)


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """
    def to(self, device):
        return self

class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """
    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset
        #print("inside DataLoaderWithTaskname ////// :", self.data_loader )

    def __len__(self):
        #print("inside DataLoaderWithTaskname ////// :", len(self.data_loader) )

        return len(self.data_loader)
    
    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """
    print("inside multitask data loader ////////////////////")
    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        print("inside MultitaskDataloader , 2 ///////// ",self.dataloader_dict)
        self.num_batches_dict = {
            task_name: len(dataloader) 
            for task_name, dataloader in self.dataloader_dict.items()
        }
        print("inside init num_batches_dict : ", self.num_batches_dict)
        self.task_name_list = list(self.dataloader_dict)
        print("inside init task_name_list : ", self.task_name_list)
        '''
        self.dataset = [None] * sum(
            len(dataloader.dataset) 
            for dataloader in self.dataloader_dict.values()
        )
        print("inside init dataset shape: ", len(self.dataset))'''

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        print("inside __iter__ task_choice_list shape : ", task_choice_list.shape)
        np.random.shuffle(task_choice_list)
        print("inside __iter__ task_choice_list_2 : ", task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader) 
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])   




class MultitaskTrainer(transformers.Trainer):

    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        print("inside get_single_train_dataloader task name ///////////////// ", task_name)
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
              train_dataset,
              batch_size=self.args.train_batch_size,
              sampler=train_sampler,
              collate_fn= data_collator,# self.data_collator.collate_batch,
            ),
        )
 
        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each 
        task Dataloader
        """
        print("inside get_train_dataloader ///////////////////////////////////////////// ")
        return MultitaskDataloader({
            task_name: self.get_single_train_dataloader(task_name, task_dataset)
            for task_name, task_dataset in self.train_dataset.items()
        })

train_dataset = {
    task_name: dataset["train"] 
    for task_name, dataset in features_dict.items()
}
print("train dataset is ", train_dataset)


trainer = MultitaskTrainer(
    model=multitask_model,
    args=transformers.TrainingArguments(
        f"callbacks_MTL_{args.model_name}_{args.add_task}_{args.learning_rate}_{args.model_path}_{args.checkpont_p}",
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        do_train=True,
        save_strategy = "epoch",
        num_train_epochs=args.epoch,
        save_total_limit = args.epoch, 
        weight_decay=args.weight_decay,
        dataloader_num_workers= 16,
        warmup_ratio=args.warmup_ratio,
        # Adjust batch size if this doesn't fit on the Colab GPU
        per_device_train_batch_size=args.batch_size,  
        #save_steps=200,
    ),
    data_collator=data_collator,
    train_dataset=train_dataset,
    #eval_dataset = eval_dataset,
)
trainer.train()

torch.save(trainer.model, f"{args.model_name}_{args.batch_size}_{args.learning_rate}.pt")
the_model = torch.load(f"{args.model_name}_{args.batch_size}_{args.learning_rate}.pt")
#print(the_model)
state_a = trainer.model.taskmodels_dict["xnli"].state_dict().__str__()
state_b = the_model.taskmodels_dict["xnli"].state_dict().__str__()
if state_a == state_b:
    print("Network not updating ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
else:
    print("network is workinhg ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
