
import os
from sklearn import metrics
from keras import callbacks

def PrintScore(true, pred, savePath=None, average='macro'):
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(os.path.join(savePath, "Result.txt"), 'w+')
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:")
    print('Acc\tF1S\tKappa\tF1_W\tF1_N1\tF1_N2\tF1_N3\tF1_R', file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (metrics.accuracy_score(true, pred),
                                                              metrics.f1_score(
                                                                  true, pred, average=average),
                                                              metrics.cohen_kappa_score(
                                                                  true, pred),
                                                              F1[0], F1[1], F1[2], F1[3], F1[4]),
          file=saveFile)
    # Classification report
    print("\nClassification report:", file=saveFile)
    print(metrics.classification_report(true, pred, target_names=[
          'Wake', 'N1', 'N2', 'N3', 'REM']), file=saveFile)
    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true, pred), file=saveFile)
    # Overall scores
    print('\n    Accuracy\t', metrics.accuracy_score(true, pred), file=saveFile)
    print(' Cohen Kappa\t', metrics.cohen_kappa_score(true, pred), file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true, pred,
                                             average=average), '\tAverage =', average, file=saveFile)
    print('   Precision\t', metrics.precision_score(true, pred,
                                                    average=average), '\tAverage =', average, file=saveFile)
    print('      Recall\t', metrics.recall_score(true, pred,
                                                 average=average), '\tAverage =', average, file=saveFile)
    # Results of each class
    print('\nResults of each class:', file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true,
                                             pred, average=None), file=saveFile)
    print('   Precision\t', metrics.precision_score(
        true, pred, average=None), file=saveFile)
    print('      Recall\t', metrics.recall_score(
        true, pred, average=None), file=saveFile)
    if savePath != None:
        saveFile.close()
    return metrics.accuracy_score(true, pred), metrics.f1_score(true, pred, average=average), metrics.cohen_kappa_score(true, pred)
