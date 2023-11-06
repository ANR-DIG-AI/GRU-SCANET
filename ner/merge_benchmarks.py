import os
from pathlib import Path
import pandas as pd
import json
import argparse


class MergeCSVFiles:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.labels = {}
        self.targeted_files = ['devel.csv',
                               'test.csv', 'train_dev.csv', 'train.csv']

    def save_json(self, response_json, file_name):
        try:
            with open(file_name, 'w', encoding='utf-8') as file:
                json.dump(response_json, file, indent=4, ensure_ascii=False)
            print(f"Data has been saved to '{file_name}' successfully.")
        except Exception as e:
            print(f"Error while saving the JSON file: {str(e)}")

    def replacement(self, label='', index=0, sequence=[]):
        output = []
        self.labels[label] = index
        assoc = {
            'O': 'O',
            'B': 'B' + str(index),
            'I': 'I' + str(index)
        }
        for s in sequence.split():
            output.append(assoc[s])
        return ' '.join(output)

    def merge(self):
        # Check if the output directory exists, if not, create it
        for target in self.targeted_files:
            index = 0
            target_merged_data = pd.DataFrame()
            for sub_directory in Path(self.input_dir).iterdir():
                if sub_directory.is_dir():
                    data = pd.read_csv(str(sub_directory)+'/'+target)
                    for i in range(data.shape[0]):
                        data.iloc[i]['NER'] = self.replacement(
                            label=str(sub_directory), index=index, sequence=data.iloc[i]['NER'])
                    target_merged_data = pd.concat(
                        [target_merged_data, data], ignore_index=True)
                index = index + 1
            output_file_name = f"{self.output_dir}{target.replace('.csv', '')}.csv"
            target_merged_data.to_csv(output_file_name, index=False)
            print(
                f"Merged {len(target_merged_data)} rows into {output_file_name}")
        self.save_json(response_json=self.labels,
                       file_name=self.output_dir+'labels.json')

# Exemple d'utilisation :
# input_directory = "../data/processed/"
# output_directory = "../data/processed/"


if __name__ == '__main__':
    def arg_manager():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", type=str,
                            default="../data/processed_0/")
        parser.add_argument("--output_path", type=str,
                            default="../data/processed_0/")
        return parser.parse_args()
    args = arg_manager()
    merger = MergeCSVFiles(args.input_path, args.output_path)
    merger.merge()
