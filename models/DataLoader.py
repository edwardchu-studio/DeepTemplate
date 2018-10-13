'''
This file is for class DataLoader, which construct the base structure of dataset
'''
import matplotlib.pyplot as plt


class DataLoader:
    '''
    DataLoader class
    '''

    def __init__(self):
        '''
        Initialization
        '''
        self.data_path = ''
        self.train_path = ''
        self.test_path = ''
        self.dataset = self.get_dataset()

    def get_dataset(self):
        '''
        Fetch dataset from the given path
        '''
        print(self.data_path)
        dataset = {'TRAIN': [], 'TEST': []}
        return dataset

    def display(self, img):
        '''
        Display if needed
        '''
        print(self.data_path)
        plt.subplots()
        plt.imshow(img)
        plt.show()
