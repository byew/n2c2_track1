import logging
import os
import pandas
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class InputFeatures:
    """A single set of features of data."""
    input_ids: int
    input_mask: list
    segment_ids: list
    label_id: int = None
    score: float = None
    guid: int = None


@dataclass
class InputExample:
    """A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        score: (Optional) float. The score of text a and b.
    """
    guid: int
    text_a: str
    text_b: str = None
    label: str = None
    score: float = None


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_score_range(self):
        """Gets the range of scores"""
        raise NotImplementedError()


class N2c2ClsProcessor(DataProcessor):
    def __init__(self, fold_id: int):
        self.fold_id = fold_id

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        file_path = os.path.join(data_dir, "train/clinicalSTS2019.train.V2.master.dataOnly.xlsx.cls_pairs.xlsx")
        return self._create_examples(file_path, train=True)

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        file_path = os.path.join(data_dir, "train/clinicalSTS2019.train.V2.master.dataOnly.xlsx.cls_pairs.xlsx")
        return self._create_examples(file_path, train=False)

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        file_path = os.path.join(data_dir, "train/clinicalSTS2019.train.V2.master.dataOnly.xlsx.cls_pairs.xlsx")
        return self._create_examples(file_path, train=False)

    def get_labels(self):
        """See base class."""
        return ['exam', 'med', 'other', 'plan', 'explained', 'timespent', 'signsymptom',
                'sdoh', 'consult', 'discharge', 'proc', 'thankyou', 'allergy']

    def _create_examples(self, file_path, train: bool):
        data = pandas.read_excel(file_path)
        if train:
            data = data[data.partition != self.fold_id]
        else:
            data = data[data.partition == self.fold_id]
        examples = [InputExample(guid=row.ID, text_a=row.sent, label=row.cat)
                    for row in data.itertuples()]

        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    max_len = 0
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            seq_len = len(tokens_a) + len(tokens_b)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            seq_len = len(tokens_a)
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        if seq_len > max_len:
            max_len = seq_len
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          guid=example.guid))

    print('Max Sequence Length: %d' % max_len)

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
