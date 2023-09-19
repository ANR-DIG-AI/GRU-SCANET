import pandas as pd
import os
import json
import re

class ParquetToCSVConverter:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def extract_words(self,text):
        words = self.clean_string(text).split()
        return [ w for w in words if w.strip()]
    
    def clean_string(self, value):
        value = re.sub(r'[\[\];\(\)\/&;_@\'\"\.\-\+]', ' ', value)
        # value = value.rstrip('. ')
        return value
    
    def save_json(self, response_json, file_name):
        try:
            with open(file_name, 'w', encoding='utf-8') as file:
                json.dump(response_json, file, indent=4, ensure_ascii=False)
            print(f"Data has been saved to '{file_name}' successfully.")
        except Exception as e:
            print(f"Error while saving the JSON file: {str(e)}")
            
    def convert_to_csv(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        vocabulary = {}
        for file_name in os.listdir(self.input_dir):
            if file_name.endswith('.parquet'):
                parquet_file_path = os.path.join(self.input_dir, file_name)

                df = pd.read_parquet(parquet_file_path)
                for text in df['text'].tolist():
                    words = self.extract_words(text)
                    for w in words: vocabulary[w]=0
                csv_file_name = os.path.splitext(file_name)[0] + '.csv'
                csv_file_path = os.path.join(self.output_dir, csv_file_name)

                df.to_csv(csv_file_path, index=False)
                print(f"Conversion from {file_name} to {csv_file_name} ended.")
        file_name = self.output_dir + 'vocabulary.json'
        print("Total count of tokens :", len(list(vocabulary.keys())))
        self.save_json(vocabulary, file_name)
        print('Vocabulary generated ! 100%')

if __name__ == "__main__":
    input_directory = "../data/corpus/"
    output_directory = "../data/corpus/" 

    converter = ParquetToCSVConverter(input_directory, output_directory)
    converter.convert_to_csv()
