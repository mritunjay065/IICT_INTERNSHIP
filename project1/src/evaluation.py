import numpy as np

def accuracy_score_scratch(y_true, y_pred):
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

def confusion_matrix_scratch(y_true, y_pred):
    """
    Returns a confusion matrix where rows are actual classes, columns are predicted classes.
    For binary classification (classes 0 and 1):
    [[TN, FP],
     [FN, TP]]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[tn, fp], [fn, tp]])

def precision_score_scratch(y_true, y_pred, pos_label=1):
    """
    Precision = TP / (TP + FP)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    
    if (tp + fp) == 0:
        return 0.0
    return tp / (tp + fp)

def recall_score_scratch(y_true, y_pred, pos_label=1):
    """
    Recall = TP / (TP + FN)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    
    if (tp + fn) == 0:
        return 0.0
    return tp / (tp + fn)

def f1_score_scratch(y_true, y_pred, pos_label=1):
    """
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    p = precision_score_scratch(y_true, y_pred, pos_label)
    r = recall_score_scratch(y_true, y_pred, pos_label)
    if (p + r) == 0:
        return 0.0
    return 2 * (p * r) / (p + r)

def classification_report_scratch(y_true, y_pred):
    """
    Generates a text report showing main classification metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    classes = np.unique(y_true)
    report = f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n"
    report += "-" * 55 + "\n"
    
    for c in classes:
        p = precision_score_scratch(y_true, y_pred, pos_label=c)
        r = recall_score_scratch(y_true, y_pred, pos_label=c)
        f1 = f1_score_scratch(y_true, y_pred, pos_label=c)
        support = np.sum(y_true == c)
        report += f"{c:<10} {p:<10.4f} {r:<10.4f} {f1:<10.4f} {support:<10}\n"
        
    report += "-" * 55 + "\n"
    acc = accuracy_score_scratch(y_true, y_pred)
    report += f"{'Accuracy':<10} {'':<10} {'':<10} {acc:<10.4f} {len(y_true):<10}\n"
    return report
