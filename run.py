import numpy as np
import os
import model as Model

import time
import argparse
import keras
from keras import callbacks

from dataloader import *
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import multi_gpu_model
from sklearn.metrics import confusion_matrix
from Utils import PrintScore

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # allocate dynamically
config.gpu_options.per_process_gpu_memory_fraction = 0.9
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

best_acc = 0
num_domain = 60

fold_string = None
model = None
test_model = None
X_test_psd = None
X_test_eog = None
y_test_psd = None
file_dir = None
filename = None
model_dir = None
overall_cm = []


class LossHistory(callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_epoch_end(self, epoch, logs={}):
        key_acc = 'Label_acc'
        key_val_acc = 'val_Label_acc'

        self.losses['epoch'].append(logs.get('Label_loss'))
        self.accuracy['epoch'].append(logs.get(key_acc))
        self.val_loss['epoch'].append(logs.get('val_Label_loss'))
        self.val_acc['epoch'].append(logs.get(key_val_acc))
        self.draw_p2in1(
            self.losses['epoch'], self.val_loss['epoch'], 'Label_loss', 'train_epoch', 'val_epoch')
        self.draw_p2in1(
            self.accuracy['epoch'], self.val_acc['epoch'], 'acc', 'train_epoch', 'val_epoch')
        global best_acc, overall_cm

        if best_acc < max(self.val_acc['epoch']):
            best_acc = max(self.val_acc['epoch'])
            y_pred = np.argmax(test_model.predict([X_test_psd, X_test_eog]), axis=1)
            y_true = np.argmax(y_test_eog, axis=1)
            cm = confusion_matrix(y_true, y_pred,
                                  labels=[0, 1, 2, 3, 4])
            print(cm)
            overall_cm[-1] = cm
            np.savetxt(filename+'.txt', cm, "%d")
            f = open(filename + '_best_acc.txt', "w")
            print(best_acc, file=f)
            f.close()
        print("acc", best_acc)

    def draw_p2in1(self, lists1, lists2, label, type1, type2):
        plt.figure()
        plt.plot(range(len(lists1)), lists1, 'r', label=type1)
        plt.plot(range(len(lists2)), lists2, 'b', label=type2)
        plt.ylabel(label)
        plt.xlabel(type1.split('_')[0]+'_'+type2.split('_')[0])
        plt.legend(loc="upper right")
        global filename
        filename = file_dir+label+'_fold'+fold_string
        plt.savefig(filename+'.jpg')
        plt.close()

    def draw_p(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(filename+'.jpg')
        plt.close()

    def end_draw(self):
        self.draw_p2in1(self.losses['epoch'], self.val_loss['epoch'], 'loss', 'train_epoch', 'val_epoch')
        self.draw_p2in1(self.accuracy['epoch'], self.val_acc['epoch'], 'acc', 'train_epoch', 'val_epoch')


def run_SENDAL(args):
    global fold_string, best_acc, model, test_model, file_dir, overall_cm, X_test_psd, X_test_eog, y_test_eog

    num_classes = 5
    classes = list(range(num_classes))
    num_folds = args.num_fold
    label_dir = args.label_dir
    data_dir_eog = args.data_dir_eog
    data_dir_psd = args.data_dir_psd
    model_dir = args.output
    file_dir = args.result_dir
    seq_len, width, height = 5, 16, 16
    num_classes = 5
    save_model = True if args.save_model else False
    gpunums = args.gpunums

    total_time = 0
    acc = np.zeros(num_folds)

    all_y_pred = []
    all_y_true = []

    if os.path.exists(file_dir) is not True:
        os.mkdir(file_dir)
    if os.path.exists(model_dir) is not True:
        os.mkdir(model_dir)

    for fold_idx in range(args.from_fold, args.to_fold + 1):
        overall_cm.append([])
        best_acc = 0
        fold_string = str(fold_idx)
        start_time_fold_i = time.time()
        logs_loss = LossHistory()
        print('Train start time of fold{} is {}'.format(
            fold_idx, start_time_fold_i))

        data_loader_eog = SeqDataLoader(
            data_dir=data_dir_eog,
            label_dir=label_dir,
            n_folds=num_folds,
            fold_idx=fold_idx,
            classes=classes
        )

        X_train_eog, y_train_eog, domain_train, X_test_eog, y_test_eog, domain_test = data_loader_eog.load_data()

        data_loader_psd = SeqDataLoader(
            data_dir=data_dir_psd,
            label_dir=label_dir,
            n_folds=num_folds,
            fold_idx=fold_idx,
            classes=classes
        )
        X_train_psd, _, _, X_test_psd, _, _ = data_loader_psd.load_data()

        model_name = f'model_fold{fold_idx:02d}_in{num_folds:02d}.h5'
        model, test_model = Model.getModel(
            num_class=num_classes,
            num_domain=num_domain,
            seq_len=seq_len,
            width=width,
            height=height,
            time_filters_nums=args.num_time_filters,
            psd_filter_nums=args.num_psd_filters,
            reduction_ratio=args.reduction_ratio,
            times=11520,
            Fs=128,
            se_activation=args.se_activation
        )

        if gpunums > 1:
            parallel_model = multi_gpu_model(model, gpus=gpunums)
            adam = keras.optimizers.Adam(
                lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            parallel_model.compile(
                optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            adam = keras.optimizers.Adam(
                lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            model.compile(
                optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks_list = [logs_loss]

        if args.early_stop == 1:
            callbacks_list.append(callbacks.EarlyStopping(
                monitor='val_Label_acc',
                patience=40))

        if save_model:
            callbacks_list.append(callbacks.ModelCheckpoint(
                filepath=model_dir + model_name,
                monitor='val_Label_acc',
                save_best_only=True,
            ))

        model.fit([X_train_psd, X_train_eog], [y_train_eog, domain_train], validation_data=([X_test_psd, X_test_eog], [y_test_eog, domain_test]),
                  epochs=args.epoch, batch_size=args.batch_size, callbacks=callbacks_list, verbose=2, shuffle=True)

        end_time_fold_i = time.time()

        y_pred = np.argmax(test_model.predict([X_test_psd, X_test_eog]), axis=1)
        y_true = np.argmax(y_test_eog, axis=1)

        all_y_pred.append(y_pred)
        all_y_true.append(y_true)

        train_time_fold_i = end_time_fold_i - start_time_fold_i
        total_time += train_time_fold_i
        logs_loss.end_draw()
        acc[fold_idx] = max(logs_loss.val_acc['epoch'])
        print('Training time of fold{} is {}'.format(fold_idx, train_time_fold_i))
        keras.backend.clear_session()

    for index in range(1, len(overall_cm)):
        overall_cm[0] += overall_cm[index]

    pred = np.concatenate(all_y_pred)
    true = np.concatenate(all_y_true)
    PrintScore(true, pred, savePath=file_dir)

    print('Training time:', total_time)
    print("Overall confusion matrix:")
    print(overall_cm[0])
    np.savetxt(os.path.join(file_dir, 'final_acc.txt', acc))


def main():
    parser = argparse.ArgumentParser(
        description='SEN-DAL on MASS-SS3 - K fold')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size (default: 64)')
    parser.add_argument('--epoch', type=int, default=100, metavar='N',
                        help='epoch (default: 100)')
    parser.add_argument('--num_fold', type=int, default=31, metavar='N',
                        help='fold num (default:31)')
    parser.add_argument('--save_model', type=int, default=1, metavar='N',
                        help='save_model (default: 1)')
    parser.add_argument('--early_stop', type=int, default=1, metavar='N',
                        help='Early Stop? (default: 1)')
    parser.add_argument('--gpunums', type=int, default=1, metavar='N',
                        help='GPU nums(default: 1)')

    parser.add_argument('--from_fold', type=int, default=0, metavar='N',
                        help='Train from fold (default: 1)')
    parser.add_argument('--to_fold', type=int, default=30, metavar='N',
                        help='Train from fold (default: 30)')

    parser.add_argument('--num_psd_filters', type=int, default=16, metavar='N',
                        help='num_psd_filters (default: 16)')
    parser.add_argument('--num_time_filters', type=int, default=64, metavar='N',
                        help='num_time_filters (default: 64)')
    parser.add_argument('--reduction_ratio', type=int, default=2, metavar='N',
                        help='reduction_ratio (default: 2)')
    parser.add_argument('--se_activation', type=str, default='sigmoid', metavar='N',
                        help='SE activation (default: sigmoid)')

    parser.add_argument('--label_dir', type=str, default='./data/label', metavar='N',
                        help='label_dir (default: ./label)')
    parser.add_argument('--data_dir_eog', type=str, default='./data/EOG', metavar='N',
                        help='data_dir_eog (default: ./data/EOG)')
    parser.add_argument('--data_dir_psd', type=str, default='./data/PSD', metavar='N',
                        help='data_dir_psd (default:./data/PSD)')
    
    parser.add_argument('--output', type=str, default='./output_model/', metavar='N',
                        help='output dir (default: ./output_model/)')            
    parser.add_argument('--result_dir', type=str, default='./result/', metavar='N',
                        help='result_dir (default: ./result/)')

    args = parser.parse_args()
    run_SENDAL(args)


if __name__ == "__main__":
    main()
