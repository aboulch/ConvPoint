import numpy as np

#==============================================
#==============================================
# STATS
#==============================================
#==============================================


def stats_overall_accuracy(cm):
    """Compute the overall accuracy.
    """
    return np.trace(cm)/cm.sum()


def stats_pfa_per_class(cm):
    """Compute the probability of false alarms.
    """
    sums = np.sum(cm, axis=0)
    mask = (sums>0)
    sums[sums==0] = 1
    pfa_per_class = (cm.sum(axis=0)-np.diag(cm)) / sums
    pfa_per_class[np.logical_not(mask)] = -1
    average_pfa = pfa_per_class[mask].mean()
    return average_pfa, pfa_per_class


def stats_accuracy_per_class(cm):
    """Compute the accuracy per class and average
        puts -1 for invalid values (division per 0)
        returns average accuracy, accuracy per class
    """
    # equvalent to for class i to
    # number or true positive of class i (data[target==i]==i).sum()/ number of elements of i (target==i).sum()
    sums = np.sum(cm, axis=1)
    mask = (sums>0)
    sums[sums==0] = 1
    accuracy_per_class = np.diag(cm) / sums #sum over lines
    accuracy_per_class[np.logical_not(mask)] = -1
    average_accuracy = accuracy_per_class[mask].mean()
    return average_accuracy, accuracy_per_class


def stats_iou_per_class(cm, ignore_missing_classes=True):
    """Compute the iou per class and average iou
        Puts -1 for invalid values
        returns average iou, iou per class
    """

    sums = (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    mask  = (sums>0)
    sums[sums==0] = 1
    iou_per_class = np.diag(cm) / sums
    iou_per_class[np.logical_not(mask)] = -1

    if mask.sum()>0:
        average_iou = iou_per_class[mask].mean()
    else:
        average_iou = 0

    return average_iou, iou_per_class


def stats_f1score_per_class(cm):
    """Compute f1 scores per class and mean f1.
        puts -1 for invalid classes
        returns average f1 score, f1 score per class
    """
    # defined as 2 * recall * prec / recall + prec
    sums = (np.sum(cm, axis=1) + np.sum(cm, axis=0))
    mask  = (sums>0)
    sums[sums==0] = 1
    f1score_per_class = 2 * np.diag(cm) / sums
    f1score_per_class[np.logical_not(mask)] = -1
    average_f1_score =  f1score_per_class[mask].mean()
    return average_f1_score, f1score_per_class
