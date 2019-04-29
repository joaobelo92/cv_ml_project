import csv
import pandas as pd
import os
import glob


class UCF101CVSParser:
    """
    This class is responsible for reading and parsing the train/val data files.
    It is used to parse the txt files to csv so they can be read with pandas
    """
    def __init__(self, path, split='01'):
        self.path = path
        self.split = split
        self.action_labels = {}
        self.train_dataset = {}

        with open(path + 'classInd.txt') as f:
            file_lines = f.readlines()
        for x in file_lines:
            label, action = x.strip('\r\n').split(' ')
            self.action_labels[action] = int(label)
        self.train_dataset = self.__load_dataset_metadata('train')
        self.val_dataset = self.__load_dataset_metadata('test')

        # print(self.train_dataset, self.val_dataset)

    def write_to_csv(self, filename, dictionary, headers):
        with open(self.path + '{}.csv'.format(filename, self.split), 'w') as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(dictionary.items())

    def add_number_frames_cvs(self, cvs_file):
        d = pd.read_csv(self.path + cvs_file + '.csv')
        d['frames'] = d['file'].apply(self.__count_number_frames)
        d.to_csv(self.path + cvs_file + '.csv', index=False)

    def __count_number_frames(self, folder):
        return len(glob.glob('{}/*.jpg'.format(os.path.join(self.path, 'jpegs_256', folder))))

    def __load_dataset_metadata(self, dataset):
        with open(self.path + '{}list{}.txt'.format(dataset, self.split)) as f:
            file_lines = f.readlines()
        dataset = {}
        for x in file_lines:
            action, filename = x.split('/')
            filename = filename.split(' ')[0].split('.')[0]
            label = self.action_labels[action]
            dataset[filename] = label

        return dataset


splitter = UCF101CVSParser('/home/joao/Datasets/ucf101/')
splitter.write_to_csv('trainlist01', splitter.train_dataset, ['file', 'label'])
splitter.write_to_csv('vallist01', splitter.val_dataset, ['file', 'label'])
splitter.write_to_csv('classInd', {v: k for k, v in splitter.action_labels.items()},
                      ['label', 'class'])
splitter.add_number_frames_cvs('trainlist01')
splitter.add_number_frames_cvs('vallist01')
