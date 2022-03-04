from abc import ABC, abstractmethod
import csv

import numpy as np

from utils.utils import load_dataset, zScoreNormalization


class BaseDataset(ABC):
    def __init__(self, name, negative_sample_labels):
        self.name = name
        self.negative_sample_labels = negative_sample_labels

    def map_labels(self, sample_labels):
        # Replace sample labels as 0 (negative) or 1 (positive)
        for i in range(len(sample_labels)):
            if sample_labels[i] in self.negative_sample_labels:
                sample_labels[i] = 0
            else:
                sample_labels[i] = 1
        return sample_labels

    @abstractmethod
    def get_samples_and_labels(self):
        pass


class LiverCancer(BaseDataset):
    def __init__(self):
        super().__init__('Liver Cancer', ['normal', 'adjacent-non_tumor'])
        self.file_name = 'data/Final_GSE25097_Matrix.csv'

    def get_samples_and_labels(self):
        samples = []
        sample_labels = []

        with open(self.file_name) as file:
            csv_reader = csv.reader(file)
            i = 0
            for row in csv_reader:
                if i == 2:
                    sample_labels = row[1:]
                if i > 5:
                    samples.append([float(x) for x in row[3:]])
                i += 1

        # Convert to Numpy Array
        samples = np.array(samples).T
        sample_labels = self.map_labels(sample_labels)

        return samples, sample_labels


class BrainCancer(BaseDataset):
    def __init__(self):
        super().__init__('Brain Cancer', ['NG'])
        self.file_name = 'data/Nutt-2003-v2_BrainCancer.csv'

    def get_samples_and_labels(self):
        samples = []
        sample_labels = []

        with open(self.file_name) as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            sample_labels = next(csv_reader)[1:]
            for row in csv_reader:
                samples.append([float(x) for x in row[1:]])

        # Convert to Numpy Array
        samples = np.array(samples).T
        sample_labels = self.map_labels(sample_labels)

        return samples, sample_labels


class ProstateCancer(BaseDataset):
    def __init__(self):
        super().__init__('Prostate Cancer', ['N'])
        self.file_name = 'data/Singh-2002_ProstateCancer.csv'

    def get_samples_and_labels(self):
        samples = []
        sample_labels = []

        with open(self.file_name) as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            sample_labels = next(csv_reader)[1:]
            for row in csv_reader:
                samples.append([float(x) for x in row[1:]])

        # Convert to Numpy Array
        samples = np.array(samples).T
        sample_labels = self.map_labels(sample_labels)

        return samples, sample_labels


class BreastCancer(BaseDataset):
    def __init__(self):
        super().__init__('Breast Cancer', ['normal', 'adjacent-non_tumor'])

    def get_samples_and_labels(self):
        normalFilePath = 'data/BC-TCGA-Normal.txt'
        tumorFilePath = 'data/BC-TCGA-Tumor.txt'

        samples, normalSampleIndexes, tumorSampleIndexes = load_dataset(normalFilePath, tumorFilePath)

        # Normalise the dataset
        samples = zScoreNormalization(samples)

        # Convert to Numpy Array
        samples = np.array(samples)

        sample_labels = [0] * (normalSampleIndexes.stop - normalSampleIndexes.start) + \
                        [1] * (tumorSampleIndexes.stop - tumorSampleIndexes.start)

        return samples, sample_labels
