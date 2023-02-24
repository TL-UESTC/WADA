import os
import operator
import random
import scipy.io as sio
import pickle as pickle
import numpy as np
import pandas as pd
from collections import OrderedDict

class DataSet_phm(object):
    '''This class is used to arrange dataset, collected and used by Lab 119 in HIT.
        module numpy, pickle and pandas should be installed before used.
        Attributes:
            name: The name of dataset with str type. And this name is used to save and load the 
                dataset with the file name as 'DataSet_' + name + '.pkl'
            index: A list contained atrributes of the dataset, so that samples can be distinguished 
                from each others by different values under same attributes.
            save_path: A string described where to save or load this dataset, and defaulted as './data/'
            dataset: A list contained samples and their attributes.
    '''
    def __init__(self,name='',index=[],save_path='./data/',dataset=[]):
        self.name = name
        self.index = index
        self.save_path = save_path
        self.dataset = dataset

    # inner function
    def _deal_condition(self,condition):
        '''
        get the index of samples whose attributes is in condition.

        Args:
            condition: A dict whose keys are the name of attributes and values are lists contained values owned by 
                samples we need.
        Return:
            A bool list whether the sample need according to condition.
        '''
        conforming_idx_all = []
        for k in condition.keys():
            value_attribute = self.get_value_attribute(k)
            conforming_idx_all.append([x in condition[k] for x in value_attribute])
        conforming_idx = [True]*len(self.dataset)
        for x in conforming_idx_all:
            conforming_idx = conforming_idx and x
        return conforming_idx

    # modify
    def reset_index(self,index):
        assert isinstance(index,list)
        self.index = index

    def add_index(self,new_attribute,new_value=None):
        '''
        Add new attribute to dataset.
        
        Args:
            new_attribute: The name of new attribute (a string).
            new_value: A list contained values appended to each sample. If the length of new_value is 1,
                then all samples will append the same new_value. Or the length of new_value should be the
                same as the number of samples, then each sample will append the corresponding value. 
                Otherwise, raise valueError.
        Return:
            None
        '''
        self.index.append(new_attribute)
        if new_value == None:
            for x in self.dataset:
                x.append(new_value)
        elif isinstance(new_value,list):
            if len(new_value) == 1:
                for i in range(len(self.dataset)):
                    self.dataset[i].append(new_value[0])
            elif len(new_value) == len(self.dataset):
                for i in range(len(self.dataset)):
                    self.dataset[i].append(new_value[i])
            else:
                raise TypeError

    def del_index(self,del_attribute):
        '''
        delete attribute and the corresponding values in each sample.
        
        Args:
            del_attribute: The name of attribute (a string).
        Return:
            None
        '''
        try:
            idx = self.index.index(del_attribute)
            for x in self.dataset:
                del(x[idx])
            self.index.remove(del_attribute)
        except ValueError:
            raise ValueError
            print('The given attribute does not exist in index, and the attributes of this dataset \
                is ', self.index)

    def append(self,append_data):
        '''
        Append samples.
        
        Args:
            append_data: A dict or a list that contain a sample, including data and attribute.
        Return:
            None
        '''
        if isinstance(append_data,dict):
            if len(append_data.keys()) <= len(self.index):
                append_data_list = []
                for x in self.index:
                    if x in list(append_data.keys()):
                        append_data_list.append(append_data[x])
                    else:
                        append_data_list.append(None)
                self.dataset.append(append_data_list)
            else:
                raise ValueError('append_data has too much attribute!')
        elif isinstance(append_data,list):
            if len(append_data) == len(self.index):
                self.dataset.append(append_data)
            else:
                raise ValueError('append_data has wrong number of attribute!')
        else:
            raise TypeError('append_data should be dict or list')

    def delete(self,condition):
        '''
        delete samples.
        
        Args:
            condition: A dict determines which samples should be delete.
        Return:
            None
        '''
        conforming_idx = self._deal_condition(condition)
        for i,x in enumerate(self.dataset):
            if conforming_idx[i]:
                self.dataset.pop(x)

    # get information or values
    def get_value_attribute(self,attribute):
        '''
        get values under the given attribute of each data
        Args:
            attribute: A str mapping the attribute of dataset.
                Return error and all attribute of dataset if the given attribute does
                not exist.
        Return:
            A list of values under the given attribute with the same order as samples in dataset.
        '''
        try:
            idx = self.index.index(attribute)
            return [x[idx] for x in self.dataset]
        except ValueError:
            raise ValueError
            print('The given attribute does not exist in index, and the attributes of this dataset \
                is ', self.index)

    def get_value(self,attribute,condition={}):
        '''
        get corresponding values.
        
        Args:
            attribute: A string describes the values returned.
            condition: A dict determines the values of which samples should be returned.
        Return:
            A list contrained values by given attribute and condition.
        '''
        conforming_idx = self._deal_condition(condition)
        idx = self.index.index(attribute)
        return [x[idx] for i,x in enumerate(self.dataset) if conforming_idx[i]]

    def get_dataset(self,condition={}):
        '''
        get corresponding dataset.
        
        Args:
            condition: A dict determines the values of which samples should be returned.
        Return:
            A DataSet contrained values by given condition.
        '''
        conforming_idx = self._deal_condition(condition)
        return DataSet_phm(name='temp',index=self.index,
                        dataset=[x for i,x in enumerate(self.dataset) if conforming_idx[i]])

    def get_random_choice(self):
        '''
        get a random sample.
        
        Args:
            None
        Return:
            A dict like {Attribute_1:Values,...}.
        '''
        r = {}
        data = random.choice(self.dataset)
        for i,k in enumerate(self.index):
            r[k] = data[i]
        return r

    def get_random_samples(self,n=1):
        '''
        get a random DataSet.
        
        Args:
            None
        Return:
            A Dataset with same index but only one sample.
        '''
        return DataSet_phm(name='temp',index=self.index,dataset=random.sample(self.dataset,n))
    
    # value process
    def normalization(self,attribute,select='std'):
        idx = self.index.index(attribute)
        for i in range(len(self.dataset)):
            if select == 'fft':
                self.dataset[i][idx] = self.dataset[i][idx] / np.max(self.dataset[i][idx])
            else:
                self.dataset[i][idx] = self.dataset[i][idx] - np.mean(self.dataset[i][idx])
                if select == 'min-max':
                    self.dataset[i][idx] = self.dataset[i][idx] / max(np.max(self.dataset[i][idx]),abs(np.min(self.dataset[i][idx])))
                elif select == 'std':
                    self.dataset[i][idx] = self.dataset[i][idx] / np.std(self.dataset[i][idx])
                else:
                    raise ValueError

    # class operation
    def shuffle(self):
        random.shuffle(self.dataset)

    def random_sample(self,n):
        if isinstance(n,str):
            if n == 'all':
                self.shuffle()
            elif n == 'half':
                self.dataset = random.sample(self.dataset,int(len(self.dataset)/2))
            else:
                raise ValueError('n should be \'all\' or \'half\'!')
        elif isinstance(n,int):
            if n >= len(self.dataset):
                self.shuffle()
            else:
                self.dataset = random.sample(self.dataset,n)
        else:
            raise TypeError('n should be int of string!')

    def dataset_filter(self,condition={}):
        conforming_idx = self._deal_condition(condition)
        self.dataset = [x for i,x in enumerate(self.dataset) if conforming_idx[i]]

    def save(self):
        '''
        Save this DataSet as .pkl file.
        
        Args:
            None
        Return:
            None
        '''
        assert self.name != ''
        assert self.save_path != ''
        pickle.dump(self, open(self.save_path + 'DataSet_' +
                                     self.name + '.pkl', 'wb'), True)
        self._save_info()
        print('dataset ', self.name, ' has benn saved\n')

    def _save_info(self):
        '''
        Save this DataSet' information as .csv file in the save_path.
        
        Args:
            None
        Return:
            None
        '''
        assert self.name != ''
        assert self.save_path != ''
        info = OrderedDict()
        for attr in self.index:
            info[attr] = self.get_value_attribute(attr)
            if isinstance(info[attr][0],np.ndarray) and len(info[attr][0])>1:
                for i,x in enumerate(info[attr]):
                    info[attr][i] = x.shape
            if not isinstance(info[attr][0],str) and len(info[attr][0]) > 2:
                for i,x in enumerate(info[attr]):
                    info[attr][i] = len(x)

        pd.DataFrame(info).to_csv(self.save_path + 'DataSet_' + self.name + 'info.csv',index=False)

    def load(self,name=''):
        '''
        Load this DataSet with name and path known, which should be given when initialize DataSet class.
        
        Args:
            name: The name of DataSet.
        Return:
            None
        '''
        if name != '':
            self.name = name
        assert self.name != ''
        assert self.save_path != ''
        full_name = self.save_path + 'DataSet_' + self.name + '.pkl'
        load_class = pickle.load(open(full_name, 'rb'))
        assert load_class.name == self.name
        assert load_class.save_path == self.save_path
        print('dataset ', self.name, ' has been load')
        self.dataset = load_class.dataset
        self.index = load_class.index

    @staticmethod
    def load_dataset(name):
        '''
        Load this DataSet with name and default path './data/'.
        
        Args:
            name: The name of DataSet.
        Return:
            DataSet
        '''
        save_path = 'data/'
        full_name = save_path + 'DataSet_' + name + '.pkl'
        load_class = pickle.load(open(full_name, 'rb'))
        print('dataset ', name, ' has been load')
        return load_class

def make_phm_dataset():
    RUL_dict = {'Bearing1_1':0,'Bearing1_2':0,
                'Bearing2_1':0,'Bearing2_2':0,
                'Bearing3_1':0,'Bearing3_2':0,
                'Bearing1_3':573,'Bearing1_4':33.9,'Bearing1_5':161,'Bearing1_6':146,'Bearing1_7':757,
                'Bearing2_3':753,'Bearing2_4':139,'Bearing2_5':309,'Bearing2_6':129,'Bearing2_7':58,
                'Bearing3_3':82}
    phm_dataset = DataSet_phm(name='phm_data',
                        index=['bearing_name','RUL','quantity','data'])
    source_path = './PHM/'
    for path_1 in ['Learning_set/', 'Test_set/']:
        bearings_names = os.listdir(source_path + path_1)
        bearings_names.sort()
        for bearings_name in bearings_names:
            file_names = os.listdir(source_path + path_1 + bearings_name + '/')
            file_names.sort()
            bearing_data = np.array([])
            for file_name in file_names:
                if 'acc' in file_name:
                    df = pd.read_csv(source_path + path_1 + bearings_name + '/'\
                                    + file_name,header=None)
                    data = np.array(df.loc[:,4:6])
                    data = data[np.newaxis,:,:]
                    if bearing_data.size == 0:
                        bearing_data = data
                    else:
                        bearing_data = np.append(bearing_data,data,axis=0)
        
            phm_dataset.append([bearings_name,RUL_dict[bearings_name],bearing_data.shape[0],bearing_data])
            print(bearings_name,'has been appended.')

    phm_dataset.save()  


if __name__ == '__main__':
    make_phm_dataset()
    dataset = DataSet_phm.load_dataset('phm_data')

    print('1')
