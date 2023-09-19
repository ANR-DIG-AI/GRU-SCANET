from compute_files import ComputeFile
import xmltodict
import pprint
import json
import multiprocessing
from datasets import load_dataset

class ExtractFeatureData:
    
    def __init__(self, input_path=''):
        print('Extract feature Data')
        self.input_path = input_path
        self.specific_tag = ['ArticleTitle', 'AbstractText']

    def recursive_search(self, dictionary, value_list):
        if not value_list:
            return None
        key = value_list[-1]
        if key in dictionary:
            if len(value_list) == 1:
                return dictionary[key]
            else:
                sub_dictionary = dictionary[key]
                sub_list = value_list[:-1]
                return self.recursive_search(sub_dictionary, sub_list)
        else:
            return None

    def process_one_file(self, percent, file=''):
        outputs = set()
        title_path = ['MedlineCitation', 'Article', 'ArticleTitle', '#text']
        title_path.reverse()
        abstract_path = ['MedlineCitation', 'Article', 'Abstract', 'AbstractText']
        abstract_path.reverse()
        abstract_path_with_text = ['MedlineCitation', 'Article', 'Abstract', 'AbstractText', '#text']
        abstract_path_with_text.reverse()
        file = file
        bad_file = None
        print('Start Percent processed : ', round(percent,2), '%')
        try:
            with open(file) as fd:
                document = xmltodict.parse(fd.read())
                for key in document['PubmedArticleSet']['PubmedArticle']:
                    article_title = self.recursive_search(key, title_path)
                    abstract_text = self.recursive_search(key, abstract_path)
                    article_text_to = self.recursive_search(key, abstract_path_with_text)
                        
                    if article_title != None :
                        # print(article_title)
                        article_title = article_title.replace('.', '').strip()
                        outputs.update(article_title.split())
                            
                    if abstract_text != None and article_text_to != None :
                        # print('------', article_text_to)
                        abstract_text = article_text_to.replace('.', '').strip()
                        outputs.update(article_text_to.split())
                            
                    if article_title != None and article_text_to != None :
                        outputs.update(article_text_to.split())
        except Exception as _:
            bad_file = file
        print('End of Percent processed : ', round(percent,2), '%')
        return outputs, bad_file

    def read_xml_content(self):
        list_files = ComputeFile(input_path=self.input_path).build_list_files()
        # print(list_files)
        outputs = set()
        bad_files = []
        total = len(list_files)
        with multiprocessing.Pool(processes=7) as pool:
            results = pool.starmap(self.process_one_file, [ (round((i/total)*100,2), list_files[i]) for i in range(0, total)])
            for output, bad_file in results:
                outputs.update(output)
                if bad_file != None : bad_files.append(bad_file)
        print('The count of corrupted files is : ' , len(bad_files))
        return list(outputs)
    
    def extract_words(self,text):
        words = text.split()
        return words

    def get_from_pmc(self):
        all_words = []
        dataset = load_dataset("pubmed")
        for example in dataset["train"]:
            title = example["title"]
            abstract = example["abstract"]
            words = self.extract_words(title)
            all_words.extend(words)
            words = self.extract_words(abstract)
            all_words.extend(words)
        return all_words
    
    def run(self):
        outputs = self.read_xml_content()
        return outputs