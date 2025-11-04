import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pandas as pd

COLUMNS = [
    "TOKEN",
    "NE-COARSE-LIT",
    "NE-COARSE-METO",
    "NE-FINE-LIT",
    "NE-FINE-METO",
    "NE-FINE-COMP",
    "NE-NESTED",
    "NEL-LIT",
    "NEL-METO",
    "RENDER",
    "SEG",
    "OCR-INFO",
    "MISC",
]
COLUMNS = [
    "TOKEN",
    "NE-COARSE-LIT",
    "NE-COARSE-METO",
    "NE-FINE-LIT",
    "NE-FINE-METO",
    "NE-FINE-COMP",
    "NE-NESTED",
    "NEL-LIT",
    "NEL-METO",
    "MISC",
]


def _read_conll(path, encoding="utf-8", sep=None, indexes=None, dropna=True):
    """
    Construct a generator to read conll items.
    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param sep: separator
    :param indexes: conll object's column indexes that are needed, if None, all columns are needed. default: None
    :param dropna: whether to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, conll item)
    """

    def parse_conll(sample):
        sample = list(map(list, zip(*sample)))
        sample = [sample[i] for i in indexes]

        for f in sample:
            if len(f) <= 0:
                raise ValueError("empty field")
        return sample

    def parse_date(line):
        # Example line: "# hipe2022:date = 1888-01-09"
        return line.split("=")[1].strip()

    with open(path, "r", encoding=encoding) as f:
        sample = []
        date = None  # Initialize date variable
        start = next(f).strip()  # Skip columns
        start = next(f).strip()

        data = []
        for line_idx, line in enumerate(f, 0):
            line = line.strip()

            if line.startswith("# hipe2022:date"):
                date = parse_date(line)  # Extract date from the metadata line
                continue

            if any(substring in line for substring in ["DOCSTART", "# id", "# "]):
                continue

            if line == "":
                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        if ["TOKEN"] not in res and ["Token"] not in res:
                            has_entities = not all(v == "O" for v in res[1])
                            data.append([line_idx, res, has_entities, date])
                    except Exception as e:
                        if dropna:
                            print(
                                f"Invalid instance which ends at line: {line_idx} has been dropped."
                            )
                            sample = []
                        else:
                            raise ValueError(
                                f"Invalid instance which ends at line: {line_idx}"
                            ) from e
            elif "EndOfSentence" in line:
                sample.append(line.split(sep) if sep else line.split())

                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        if ["TOKEN"] not in res and ["Token"] not in res:
                            has_entities = not all(v == "O" for v in res[1])
                            data.append([line_idx, res, has_entities, date])
                    except Exception as e:
                        if dropna:
                            print(
                                f"Invalid instance which ends at line: {line_idx} has been dropped."
                            )
                            sample = []
                        else:
                            raise ValueError(
                                f"Invalid instance which ends at line: {line_idx}"
                            ) from e
            else:
                sample.append(line.split(sep) if sep else line.split())

        if len(sample) > 0:
            try:
                res = parse_conll(sample)
                if ["TOKEN"] not in res and ["Token"] not in res:
                    has_entities = not all(v == "O" for v in res[1])
                    data.append([line_idx, res, has_entities, date])
            except Exception as e:
                if dropna:
                    return
                print(f"Invalid instance ends at line: {line_idx}")
                raise e

        return data


def export_entity_statistics(dataset, split_name, output_path, label_map):
    # Create structure: {(split, year, label): count}
    stats = defaultdict(int)

    for i in range(len(dataset)):
        year = int(dataset.dates[i].split("-")[0])
        token_labels = dataset.token_targets_dict["NE-COARSE-LIT"][i]
        # inverse label map
        label_map = dataset.get_inverse_label_map()
        token_labels = [label_map["NE-COARSE-LIT"][label] for label in token_labels]

        for label in token_labels:

            entity_type = label.split("-")[-1].upper() if label != "O" else "O"
            if 'B-' in label:
                stats[(split_name, year, entity_type)] += 1

    # Convert to DataFrame
    rows = [
        {"split": split, "year": year, "entity_type": entity, "count": count}
        for (split, year, entity), count in stats.items()
    ]
    df_stats = pd.DataFrame(rows)
    df_stats = df_stats.sort_values(by=["split", "year", "entity_type"])
    df_stats.to_csv(output_path, sep="\t", index=False)
    return df_stats


class NewsDataset(Dataset):
    def __init__(
            self, tsv_dataset, tokenizer, max_len, label_map={}, filter_labels=False
    ):
        """
        Initializes a dataset in IOB format.
        :param tsv_dataset: tsv filename of the train/test/dev dataset
        :param tokenizer: the LM tokenizer
        :param max_len: the maximum sequence length, can be 512 for BERT-based LMs
        :param test: if it is the test dataset or not - can be disregarded for now
        :param label_map: the label map {0: 'B-pers'}
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_name_or_path = tokenizer.name_or_path
        self.filter_labels = filter_labels

        self.lowercase = False
        if "uncased" in self.model_name_or_path:
            self.lowercase = True

        self.tsv_dataset = tsv_dataset
        self.phrases = list(
            _read_conll(
                tsv_dataset,
                encoding="utf-8",
                sep="\t",
                indexes=list(range(len(COLUMNS))),
                dropna=True,
            )
        )

        self.tokens = [item[1][0] for item in self.phrases]

        # Extract NE columns as tasks
        self.ne_tasks = [col for col in COLUMNS if col.startswith("NE-")]
        self.token_targets_dict = {task: [] for task in self.ne_tasks}
        # ['NE-COARSE-LIT', 'NE-COARSE-METO', 'NE-FINE-LIT', 'NE-FINE-METO', 'NE-FINE-COMP', 'NE-NESTED']

        if self.filter_labels:
            # We want to keep only NE-COARSE-LIT and NE-FINE-COMP
            # TODO: so here we filter the labels of NE-FINE-COMP to keep only certain labels
            # but will take it out
            self.only_comp = ['comp.function', 'comp.name', 'comp.title']
            self.only_coarse = ['pers', 'loc', 'org']
            self.ne_tasks = ["NE-COARSE-LIT", "NE-FINE-COMP"]
            for item in self.phrases:
                for idx, task in enumerate(self.ne_tasks):
                    if task in self.ne_tasks:
                        if task == "NE-COARSE-LIT":
                            print('coarse', item[1][1])
                            item[1][idx + 1] = [
                                label if any(
                                    coarse in label for coarse in self.only_coarse
                                ) else "O"
                                for label in item[1][1]
                            ]
                            self.token_targets_dict[task].append(item[1][idx + 1])
                        elif task == 'NE-FINE-COMP':
                            print(item[1][5])
                            item[1][idx + 1] = [
                                label if any(comp in label for comp in self.only_comp) else "O"
                                for label in item[1][5]
                            ]
                            self.token_targets_dict[task].append(item[1][idx + 1])
                        else:
                            self.token_targets_dict[task].append(item[1][idx + 1])
        else:
            for item in self.phrases:
                for idx, task in enumerate(self.ne_tasks):
                    self.token_targets_dict[task].append(item[1][idx + 1])

        self.label_map = label_map
        self._build_label_map()
        # print(f"Label map in dataset: {self.label_map}")

    def _build_label_map(self):
        unique_token_labels = {task: set() for task in self.ne_tasks}
        for task, labels in self.token_targets_dict.items():
            for label_seq in labels:
                unique_token_labels[task].update(label_seq)

        for task, labels in unique_token_labels.items():
            if task not in self.label_map:
                self.label_map[task] = {}
            current_map = self.label_map[task]
            missed_labels = labels - set(current_map.keys())

            num_labels = len(current_map)
            for i, missed_label in enumerate(missed_labels):
                current_map[missed_label] = num_labels + i

        for task in self.ne_tasks:
            self.token_targets_dict[task] = [
                [self.label_map[task][label] for label in label_seq]
                for label_seq in self.token_targets_dict[task]
            ]

    def __len__(self):
        return len(self.phrases)

    def get_filename(self):
        return self.tsv_dataset

    def get_label_map(self):
        return self.label_map

    def get_inverse_label_map(self):
        return {
            task: {v: k for k, v in label_map.items()}
            for task, label_map in self.label_map.items()
        }

    def tokenize_and_align_labels(self, sequence, token_targets_dict):
        if self.lowercase:
            sequence = [word.lower() for word in sequence]
        tokenized_inputs = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            is_split_into_words=True,
            return_token_type_ids=True,
        )
        labels_dict = {task: [] for task in token_targets_dict}
        label_all_tokens = False
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                for task in labels_dict:
                    labels_dict[task].append(-100)
            elif word_idx != previous_word_idx:
                for task in labels_dict:
                    labels_dict[task].append(token_targets_dict[task][word_idx])
            else:
                for task in labels_dict:
                    if label_all_tokens:
                        labels_dict[task].append(token_targets_dict[task][word_idx])
                    else:
                        labels_dict[task].append(-100)
            previous_word_idx = word_idx

        tokenized_inputs["token_targets"] = labels_dict
        return tokenized_inputs

    def tokenize_text(self, sequence):
        tokenized_inputs = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            is_split_into_words=True,
            return_token_type_ids=True,
        )
        return tokenized_inputs

    def __getitem__(self, index):
        sequence = self.tokens[index]
        token_targets_dict = {
            task: self.token_targets_dict[task][index] for task in self.ne_tasks
        }

        encoding = self.tokenize_and_align_labels(sequence, token_targets_dict)

        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        token_type_ids = torch.tensor(encoding["token_type_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)
        token_targets = {
            task: torch.tensor(encoding["token_targets"][task], dtype=torch.long)
            for task in encoding["token_targets"]
        }

        assert input_ids.shape == attention_mask.shape
        for task in token_targets:
            assert token_targets[task].shape == input_ids.shape
        return {
            "sequence": " ".join(sequence),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_targets": token_targets,
            "token_type_ids": token_type_ids,
        }

    def get_info(self):
        num_token_labels_dict = {
            task: len(self.label_map[task]) for task in self.ne_tasks
        }
        return len(set([item[2] for item in self.phrases])), num_token_labels_dict, self.num_years
