from socket import VM_SOCKETS_INVALID_VERSION
import requests
import json
from datasets import load_dataset


class DownloadCorpus:
    
    def __init__(self):
        print('Downloading of data started 0%')
    
    def extract_words(self,text):
        words = text.split()
        return words

    def get_from_pmc(self):
        vocabulary = {}
        dataset = load_dataset("pmc/open_access")
        for example in dataset["train"]:
            title = example["title"]
            abstract = example["abstract"]
            words = self.extract_words(title)
            for w in words: vocabulary[w]=0
            words = self.extract_words(abstract)
            for w in words: vocabulary[w]=0
        return vocabulary

    def save_json(self, response_json, file_name):
        try:
            with open(file_name, 'w', encoding='utf-8') as file:
                json.dump(response_json, file, indent=4, ensure_ascii=False)
            print(f"Data has been saved to '{file_name}' successfully.")
        except Exception as e:
            print(f"Error while saving the JSON file: {str(e)}")
    
    def run(self):
        outputs = self.get_from_pmc()
        file_name = '../data/corpus/corpus.json'
        self.save_json(outputs, file_name)
        print('Downloading of data started 100%')
        return outputs

DownloadCorpus().run()