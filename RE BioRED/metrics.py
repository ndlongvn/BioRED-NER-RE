try:
    from sklearn.metrics import f1_score, precision_score, recall_score

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn

if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        return {
            "acc": acc,
            "f1_micro": f1_score(y_true=labels, y_pred=preds, average="micro"),
            "f1_macro": f1_score(y_true=labels, y_pred=preds, average="macro"),
            "precision_macro": precision_score(y_true=labels, y_pred=preds, average="macro"),
            "precision_micro": precision_score(y_true=labels, y_pred=preds, average="micro"),
            "recall_macro": recall_score(y_true=labels, y_pred=preds, average="macro"),
            "recall_micro": recall_score(y_true=labels, y_pred=preds, average="micro"),
        }
    def acc_f1_prec_rec(preds, labels):
        acc = simple_accuracy(preds, labels)
        return {
            "acc": acc,
            "f1_micro": f1_score(y_true=labels, y_pred=preds, average="micro"),
            "f1_macro": f1_score(y_true=labels, y_pred=preds, average="macro"),
            "f1": f1_score(y_true=labels, y_pred=preds, average=None)[1],
            "precision": precision_score(y_true=labels, y_pred=preds, average=None)[1],
            "recall": recall_score(y_true=labels, y_pred=preds, average=None)[1],
            "precision_macro": precision_score(y_true=labels, y_pred=preds, average="macro"),
            "precision_micro": precision_score(y_true=labels, y_pred=preds, average="micro"),
            "recall_macro": recall_score(y_true=labels, y_pred=preds, average="macro"),
            "recall_micro": recall_score(y_true=labels, y_pred=preds, average="micro"),
        }


    def compute_metrics(preds, labels): # mean of accuracy and f1
        assert len(preds) == len(labels)
        return acc_and_f1(preds, labels)
    def compute_metrics_f1(preds, labels):
        assert len(preds) == len(labels)
        return acc_f1_prec_rec(preds, labels)

