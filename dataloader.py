import os
import numpy as np
from keras.utils import to_categorical

class DomainGenerator():
    '''
    Domin Generator
    '''
    # Initializate
    def __init__(self, n_domain):
        self.n_domain = n_domain

    # Get i-th fold
    def getDomains(self, len_list, data_type='train'):
        assert data_type in ('train', 'test'), 'Wrong data type!'
        domain_list = []

        for domain_idx, cur_domain_length in enumerate(len_list):
            tmp_domain = np.zeros((cur_domain_length, self.n_domain), dtype=int)
            if data_type == 'train':
                tmp_domain[:, domain_idx] = 1
            elif data_type == 'test':
                pass
            domain_list.append(tmp_domain)
        return domain_list

class SeqDataLoader():
    def __init__(self, data_dir, label_dir, n_folds, fold_idx, classes):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.classes = classes


    def _load_npy_list_files(self, data_files, label_files):
        data = []
        labels = []
        length = []
        for data_name, label_name in zip(data_files, label_files):
#             print ("Loading {} {} ...".format(data_name,label_name))
            tmp_data = np.load(data_name)
            tmp_labels = np.load(label_name)
            tmp_labels = to_categorical(tmp_labels, num_classes=self.classes)
            tmp_labels = tmp_labels.astype(int)

            assert len(tmp_labels) == len(tmp_data), 'X and y length unmatched!'
            tmp_length = len(tmp_labels)

            data.append(tmp_data)
            labels.append(tmp_labels)
            length.append(tmp_length)
        return data, labels, length

    def print_n_samples_each_class(self, labels, classes):
        class_dict = dict(zip(range(len(classes)), classes))
        unique_labels = np.unique(labels)
        for c in unique_labels:
            n_samples = len(np.where(labels == c)[0])
            print("{}: {}".format(class_dict[c], n_samples))

    def load_data(self, shuffle=False):
        data_files = []
        for f in os.listdir(self.data_dir):
            if ".npy" in f:
                data_files.append(os.path.join(self.data_dir, f))

        label_files = []
        for f in os.listdir(self.label_dir):
            if ".npy" in f:
                label_files.append(os.path.join(self.label_dir, f))

        data_files.sort(key=lambda x: (len(x), x))
        label_files.sort(key=lambda x: (len(x), x))

        r_permute = np.random.permutation(len(data_files))
        filename = "shuffle.npz"
        if (os.path.isfile(filename)):
            with np.load(filename) as f:
                r_permute = f["inds"]
        else:
            save_dict = {
                "inds": r_permute,
            }
            np.savez(filename, **save_dict)

        data_files = np.asarray(data_files)[r_permute]
        label_files = np.asarray(label_files)[r_permute]
        traindata_files = np.array_split(data_files, self.n_folds)
        trainlabel_files = np.array_split(label_files, self.n_folds)
        subjectdata_files = traindata_files[self.fold_idx]
        subjectlabel_files = trainlabel_files[self.fold_idx]
        traindata_files = list(set(data_files) - set(subjectdata_files))
        trainlabel_files = list(set(label_files) - set(subjectlabel_files))
        traindata_files.sort(key=lambda x: (len(x), x))
        trainlabel_files.sort(key=lambda x: (len(x), x))

        # Load training and validation sets
        print("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print("Load training set:")
        data_train, label_train, length_train = self._load_npy_list_files(
            traindata_files, trainlabel_files)
        print(" ")
        print("Load Test set:")
        data_test, label_test, length_test = self._load_npy_list_files(
            subjectdata_files, subjectlabel_files)

        # Display data and label statistics
        print(" ")
        print("Training set: n_subjects={}".format(len(data_train)))
        n_train_samples = 0
        for d in data_train:
            n_train_samples += d.shape[0]
        print("Number of samples = {}".format(n_train_samples))
        self.print_n_samples_each_class(np.concatenate(label_train), self.classes)
        print(" ")
        print("Test set: n_subjects = {}".format(len(data_test)))
        n_test_samples = 0
        for d in data_test:
            #print (d.shape)
            n_test_samples += d.shape[0]
        print("Number of samples = {}".format(n_test_samples))
        self.print_n_samples_each_class(np.concatenate(label_test), self.classes)
        print()

        # Generate domain
        dg = DomainGenerator(len(length_train))
        domain_train = dg.getDomains(len_list=length_train, data_type='train')
        domain_test = dg.getDomains(len_list=length_test, data_type='test')


        # Stack list as ndarray

        data_train = np.vstack(data_train)
        label_train = np.concatenate(label_train)
        domain_train = np.concatenate(domain_train)

        data_test = np.vstack(data_test)
        label_test = np.concatenate(label_test)
        domain_test = np.concatenate(domain_test)

        print(data_train.shape)
        print(label_train.shape)
        print(domain_train.shape)
        if shuffle is True:
            # training data
            np.random.seed(0)
            permute = np.random.permutation(len(label_train))
            data_train = data_train[permute]
            label_train = label_train[permute]
            domain_train = domain_train[permute]


            # testing data
            np.random.seed(0)
            permute = np.random.permutation(len(label_test))
            data_test = data_test[permute]
            label_test = label_test[permute]
            domain_test = domain_test[permute]
        
        return data_train, label_train, domain_train, data_test, label_test, domain_test