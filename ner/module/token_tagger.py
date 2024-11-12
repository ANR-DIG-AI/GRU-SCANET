
import tiktoken
import jaro


class SentenceTokenTagger:

    def __init__(self, sequence=None, tagger=None):
        self.sequence = sequence.split(" ")
        self.tagger = tagger.split(" ")
        self.tokenizer_models = ["cl100k_base"]  # , "p50k_base", "r50k_base"]
        self.pair_sequence_tags = [(s, t)
                                   for s, t in zip(self.sequence, self.tagger)]

    def update_tagging(self, data=[]):
        output = []
        b_founded = False
        for token, id, tag in data:
            if not b_founded and 'B' in tag:
                b_founded = True
                output.append((token, id, tag))
                continue
            elif b_founded and 'B' in tag:
                output.append((token, id, tag.replace('B', 'I')))
                continue
            output.append((token, id, tag))
            b_founded = False
        return output

    def similarity(self, value1='', value2=''):
        return jaro.jaro_winkler_metric(f'{value1}', f'{value2}')

    def encode_sequence(self, sequence):
        output = []
        sequence = " ".join(sequence)
        for encoding_name in self.tokenizer_models:
            encoding = tiktoken.get_encoding(encoding_name)
            token_integers = encoding.encode(sequence)
            token_strings = [encoding.decode_single_token_bytes(
                token).decode("utf-8") for token in token_integers]
            for id, token in zip(token_integers, token_strings):
                output.append((token, id))
        return output

    def tokenize_sequence(self):
        output = []
        ids = []
        values = self.encode_sequence(self.sequence)
        for token, id in values:
            output.append(token)
            ids.append(id)
        return output, ids

    def run(self):
        output = []
        if self.sequence is not None:
            pair_token_id = self.encode_sequence(self.sequence)
            stop = 0
            for i in range(len(self.pair_sequence_tags)):
                word, tag = self.pair_sequence_tags[i]
                sim = 0
                sub_sequence = ''
                j = stop
                while j < len(pair_token_id):
                    token, id = pair_token_id[j]
                    sub_sequence = sub_sequence + token
                    _sim = self.similarity(value1=sub_sequence, value2=word)
                    if _sim > sim:
                        output.append((token, id, tag))
                        sim = _sim
                    else:
                        if _sim == sim:
                            output.append((token,  id, tag))
                        else:
                            break
                    j += 1
                stop = j
        output = self.update_tagging(data=output)
        tokens = [token for token, _, _ in output]
        tags = [tag for _, _, tag in output]
        return tokens, tags
