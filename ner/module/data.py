import numpy as np
import pandas as pd
from .token_tagger import SentenceTokenTagger


class DataProcessing:
    """
    Static class to process raw data
    """

    @staticmethod
    def read_csv(csv, is_test=False):
        """Read data from csv file, generate sentences and name entities respectively

        Args:
            csv (str): csv file path
            is_test (bool): whether it is processing test set

        Returns:
            list of sentences, list of entities
        """

        df = pd.read_csv(csv)
        sentence = df['Sentence'].tolist()
        entity = df['NER'].tolist()

        if is_test:
            tokens, _ = SentenceTokenTagger(
                sentence, None).tokenize_sequence()
            return tokens, None
            # return [s.split() for s in sentence], None
        else:
            # sens = [s.split() for s in sentence]
            # ents = [e.split() for e in entity]
            sens = []
            ents = []
            for se, en in zip(sentence, entity):
                tokens, tags = SentenceTokenTagger(se, en).run()
                sens.append(tokens)
                ents.append(tags)
            for se, en in zip(sens, ents):
                assert len(se) == len(en)

            return sens, ents

    @staticmethod
    def read_text_file(file_name):
        try:
            with open(file_name, 'r', encoding='utf-8') as file:
                content = file.read()
                words = content.split()  # Split the content into words using space as a separator
                return words
        except FileNotFoundError:
            print("The specified file was not found.")
            return []

    @staticmethod
    def build_lookup(tokens, **extra_signs):
        """Build lookup table for given tokens and possible extra signs

        Args:
            tokens (list[str]): unique tokens list
            **extra_signs (dict[str, int], optional): some extra signs such as <oov>, <pad>

        Returns:
            lookup table (dict[str, int])
        """

        lookup = {sign: idx for sign, idx in extra_signs.items()}
        for token in tokens:
            if token in lookup.keys():
                raise KeyError(
                    'Duplicate token {} found in tokens or extra_signs'.format(token))

            lookup[token] = len(lookup)

        return lookup

    @staticmethod
    def pad_sequence(sequences, max_len, padding_value):
        padded = []
        for seq in sequences:
            if len(seq) < max_len:
                padded.append(seq + [padding_value] * (max_len - len(seq)))
            else:
                padded.append(seq[:max_len])
        return padded


class DataLoader:

    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

        self._num_data = self.x.shape[0]
        self._indexes = np.arange(self._num_data)

    def gen_batch(self, batch_size, shuffle=True):
        if shuffle:
            np.random.shuffle(self._indexes)

        i = 0
        while True:
            if i + batch_size >= self._num_data:
                yield self.x[self._indexes[i:]], self.y[self._indexes[i:]]
                break

            else:
                yield self.x[self._indexes[i:i + batch_size]], self.y[self._indexes[i:i + batch_size]]
                i += batch_size

    def __len__(self):
        return self._num_data
