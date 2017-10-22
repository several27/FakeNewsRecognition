from multiprocessing import Pool
from typing import List
from time import time
from tqdm import tqdm
import os.path
import json
import re
import os
import gc

import nltk
import spacy


def tagger_batch_perform(text):
    tagger = Tagger(text)
    tagger.perform()
    return tagger.words


class Tagger:
    tags = {'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    # tags = {'NN', 'NNS', 'NNP'}

    text = ""
    original_words = []
    words = []
    tokens = []

    functions = []

    nlp = None

    def __init__(self, text):
        self.text = text

        self.functions = [
            self.expand_abbreviations,
            # self.lemma,
            self.stem,
            self.clean,
        ]

    def perform(self):
        # self.words = self.extract_words(self.text)
        #
        # for function in self.functions:
        #     self.words = function(self.words)

        if Tagger.nlp is None:
            Tagger.preload_nlp()

        tokens = Tagger.nlp(self.text)

        self.sentences = []

        self.tokens = [token for token in tokens
                       if not token.is_stop and not token.is_punct and token.tag_ in self.tags]
                       # and not token.is_digit
                       # and token.lemma_.isalnum() and len(token.lemma_) > 3 and token.tag_ in self.tags]

        self.original_words = [token.orth_ for token in self.tokens]
        self.words = [token.lemma_ for token in self.tokens]
        
    @staticmethod
    def batch_perform(texts, processes=8, chunksize=100 * 1000):
        if Tagger.nlp is None:
            Tagger.preload_nlp()

        bar = tqdm(total=len(texts))
        words_matrix = []
        pool = Pool(processes=processes, maxtasksperchild=1)
        for words in pool.imap(tagger_batch_perform, texts, chunksize=chunksize):
            words_matrix.append(words)
            bar.update()

        return words_matrix

    #         bar = tqdm(total=len(texts))
    #         words_matrix = []
    #         for doc in Tagger.nlp.pipe(texts, n_threads=8):
    #             words_matrix.append([token for token in doc
    #                                  if not token.is_stop and not token.is_punct and not token.is_digit
    #                                  and token.lemma_.isalnum() and len(token.lemma_) > 3 and token.tag_ in Tagger.tags])

    #             bar.update()

    #         return words_matrix

    @staticmethod
    def batch_perform_big_json(json_path, processes=8, chunk_size=100 * 1000):
        directory_path = os.path.dirname(json_path)
        all_files = os.listdir(directory_path)
        contains_chunk = sum([1 for file in all_files if file.startswith(os.path.basename(json_path))])
        
        chunks_paths = []
        if contains_chunk > 0:
            chunks_paths = [os.path.join(directory_path, file) for file in all_files 
                            if file.startswith(os.path.basename(json_path)) and re.search(r'chunk\_[0-9]+$', file)]
            print('Found chunks paths')
            print(chunks_paths)
        else:
            with open(json_path, 'r') as json_file_input:
                print('Loading big json')
                big_json = json.load(json_file_input)
                if not isinstance(big_json, list):
                    raise ValueError('Supplied JSON does not contain an array')

                print('Splitting it into smaller files')

                chunk_count = 0
                chunks_paths = []
                for json_chunk in Tagger._chunks(big_json, processes * chunk_size):
                    chunk_path = '%s.chunk_%s' % (json_path, chunk_count)
                    if not os.path.isfile(chunk_path):
                        with open(chunk_path, 'w') as json_file_chunk_output:
                            json.dump(json_chunk, json_file_chunk_output)

                    chunk_count += 1
                    chunks_paths.append(chunk_path)

                del big_json

        print('Tagging')
        for chunk_path in chunks_paths:                    
            output_chunk_path = chunk_path + '.out'
            
            if os.path.isfile(output_chunk_path):
                continue
                
            with open(chunk_path, 'r') as json_file_chunk_input:
                chunk = json.load(json_file_chunk_input)
                result = Tagger.batch_perform(chunk, processes)
                

                with open(output_chunk_path, 'w') as json_file_result_chunk_output:
                    json.dump(result, json_file_result_chunk_output)
                    
                del chunk
                del result

    @staticmethod
    def _chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
            
    @staticmethod
    def _load_chunked_json(json_path):
        directory_path = os.path.dirname(json_path)
        file_name = os.path.basename(json_path)
        files = Tagger._natural_sort(os.listdir(directory_path))
        
        data = []
        for file in files: 
            if not file.startswith(file_name) or not re.search(r'chunk\_[0-9]+\.out$', file):
                continue
            
            path = os.path.join(directory_path, file)
            print(path)
            with open(path, 'r') as chunk_input:
                data += json.load(chunk_input)
            
            gc.collect()
                
        return data
    
    @staticmethod
    def _merge_chunked_json(json_path):
        directory_path = os.path.dirname(json_path)
        file_name = os.path.basename(json_path)
        files = Tagger._natural_sort(os.listdir(directory_path))
        
        with open(json_path + '.out', 'w') as merged_json_output: 
            for file in files: 
                if not file.startswith(file_name) or not re.search(r'chunk\_[0-9]+\.out$', file):
                    continue

                path = os.path.join(directory_path, file)
                print(path)
                with open(path, 'r') as chunk_input:
                    json_chunk = json.load(chunk_input)
                    for element in json_chunk:
                        merged_json_output.write(json.dumps(element) + '\n')
                        
                    merged_json_output.flush()
                    
                gc.collect()
    
    @staticmethod
    def _iterate_over_json_chunks(json_path):
        directory_path = os.path.dirname(json_path)
        file_name = os.path.basename(json_path)
        files = Tagger._natural_sort(os.listdir(directory_path))
        
        for file in files: 
            if not file.startswith(file_name) or not re.search(r'chunk\_[0-9]+\.out$', file):
                continue

            path = os.path.join(directory_path, file)
            print(path)
            with open(path, 'r') as chunk_input:
                if count > 0:
                    merged_json_output.write(',')
                
                chunk_contents = chunk_input.read()
                merged_json_output.write(chunk_contents[1:-1])
                merged_json_output.flush()
                count += 1
                
            gc.collect()
    
    @staticmethod
    def _iterate_over_merged_json(file_path):
        data = []
        with open(file_path, 'r') as merged_input: 
            for line in merged_input: 
                data.append(json.loads(line))
            
        return data
        
    
    @staticmethod
    def _natural_sort(l): 
        convert = lambda text: int(text) if text.isdigit() else text.lower() 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(l, key = alphanum_key)

    @staticmethod
    def preload_nlp():
        Tagger.nlp = spacy.load('en')

    @staticmethod
    def extract_words(text: str) -> List[str]:
        if Tagger.nlp is None:
            Tagger.nlp = spacy.load('en')

        tokens = Tagger.nlp(text)

        return [token.string.strip() for token in tokens if token.tag_ in Tagger.tags]

    @staticmethod
    def expand_abbreviations(words: List[str]) -> List[str]:
        # @TODO
        return words

    @staticmethod
    def lemma(words: List[str]) -> List[str]:
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in words]

    @staticmethod
    def stem(words: List[str]) -> List[str]:
        """
        Much faster that lemma, however only cuts words entries -> entri instead of entry
        """
        porter = nltk.PorterStemmer()
        return [porter.stem(word) for word in words]

    @staticmethod
    def clean(words: List[str]) -> List[str]:
        return [word.lower() for word in words if len(word) > 3]
