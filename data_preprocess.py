# -*- encoding: utf-8 -*-
'''
@File    :   data_preprocess.py
@Time    :   2020/07/05 19:29:57
@Author  :   Luo Jianhui 
@Version :   1.0
@Contact :   kid1412ljh@outlook.com
'''

# here put the import lib
import numpy as np
import pandas as pd
import string
from functools import reduce

import re
from tqdm import tqdm
from BERTweet.TweetNormalizer import normalizeTweet

import torch
from torch.utils.data import TensorDataset, random_split
from transformers import AlbertTokenizer


class PreProcess:

    def __init__(self, path):
        self.path = path
        self.result = self.extract_all_punct()
        self.tokenizer = AlbertTokenizer.from_pretrained('models')


    def load_data(self):
        data = pd.read_csv(self.path)
        data.dropna(inplace=True)
        return data

    @staticmethod
    def processrow(row):
        row = normalizeTweet(row)
        return row

    @staticmethod
    def extract_punct(x):
        x = x.split('\n')
        pattern = re.compile(r'[^A-za-z\d\s]+')
        result = pattern.findall(' '.join(x))
        result = set(result)
        return result

    def extract_all_punct(self):
        data = self.load_data()
        qb = data['question_body'].values.tolist()
        result = list(map(self.extract_punct, qb))
        result = reduce(lambda x, y: x.union(y), result)
        common_punct = [",", ".", "!", ":", "?", "'", ";", "/", '"']
        for punct in common_punct:
            result.remove(punct)
        result = list(result)
        return result


    def clean_data(self, row):
        row = row.split('\n')
        for i in range(1, len(row) - 1):
            row[i] = row[i].strip()
            for punct in self.result:
                if punct in row[i]:
                    row[i] = ''
                    break
        return ' '.join([item.strip() for item in row if item != ''])


    def load_preprocessed_data(self):

        data = self.load_data()
        
        data['question_body'] = data['question_body'].apply(self.clean_data)
        data['answer'] = data['answer'].apply(self.clean_data)

        data['question_title'] = data['question_title'].apply(
            lambda x: self.processrow(x))
        data['question_body'] = data['question_body'].apply(
            lambda x: self.processrow(x))
        data['answer'] = data['answer'].apply(lambda x: self.processrow(x))
        data['question'] = data['question_title'] + ' ' + data['question_body']
        X = data[['question', 'answer']]
        Y = data.loc[:, data.dtypes == np.float64]

        return X, Y


    def tokenize(self):
        """Tokenize all of the sentences and map the tokens to thier word IDs.
        """

        X, Y = self.load_preprocessed_data()
        questions = X['question'].values.tolist()
        answers = X['answer'].values.tolist()
        labels = Y.values

        input_ids_list = []
        token_ids_list = []
        mask_list = []

        for i in range(len(questions)):
            encoded_input = self.tokenizer(questions[i],
                                            answers[i],
                                            padding='max_length', 
                                            truncation='longest_first',
                                            max_length=512,
                                            return_tensors="pt")

            input_ids_list.append(encoded_input['input_ids'])
            token_ids_list.append(encoded_input['token_type_ids'])
            mask_list.append(encoded_input['attention_mask'])

        input_ids = torch.cat(input_ids_list, dim=0)
        token_type_ids = torch.cat(token_ids_list, dim=0)
        attention_mask = torch.cat(mask_list, dim=0)
        
        labels = torch.tensor(labels, dtype=torch.double)

        # Save vocabulary
        # output_dir = "./models/"
        # tokenizer.save_vocabulary(output_dir)

        return input_ids, token_type_ids, attention_mask, labels


    def split_data(self):

        input_ids, token_type_ids, attention_mask, labels = self.tokenize()
        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, token_type_ids, attention_mask, labels)

        # Create a 90-10 train-validation split.

        # Calculate the number of samples to include in each set.
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        return train_dataset, val_dataset


if __name__ == '__main__':
    path = 'google-quest-challenge/train.csv'
    PreProcess(path).split_data()