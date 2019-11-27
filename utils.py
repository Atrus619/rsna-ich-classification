import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats as ss
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, classification_report, hamming_loss, roc_curve, auc
import warnings
from classes.ResNeXt import ResNeXt
from scipy import interp
from itertools import cycle


def get_pretty_time(duration, num_digits=2):
    # Duration is assumed to be in seconds. Returns a string with the appropriate suffix (s/m/h)
    if duration > 60**2:
        return str(round(duration / 60**2, num_digits)) + 'h'
    if duration > 60:
        return str(round(duration / 60, num_digits)) + 'm'
    else:
        return str(round(duration, num_digits)) + 's'


def train_val_split(csv_file, test_prop=0.25, random_state=None):
    labels = pd.read_csv(csv_file, index_col=0)

    train, val = train_test_split(labels, test_size=test_prop, random_state=random_state, stratify=labels[['epidural', 'intraparenchymal', 'subarachnoid']])

    return train.index, val.index


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def get_chi2_p_value(x, y):
    confusion_matrix = pd.crosstab(x, y)
    return ss.chi2_contingency(confusion_matrix)[1]


def load_model_trainer(path):
    with open(path, 'rb') as f:
        return pkl.load(f)


def bar_plot_condition_occurrences(data):
    categories = list(data.iloc[:, 1:].columns.values)
    sns.set(font_scale=2)
    plt.figure(figsize=(15, 8))

    ax = sns.barplot(categories, data.iloc[:, 1:].sum().values)

    plt.title('Number of occurrences of each condition', fontsize=24)
    plt.xlabel('Condition', fontsize=18)
    plt.ylabel('Number of occurrences', fontsize=18)

    rects = ax.patches
    labels = data.iloc[:, 1:].sum().values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom', fontsize=18)
    plt.show()


def bar_plot_condition_co_occurrences(data):
    rowSums = data.iloc[:, 1:].sum(axis=1)
    multiLabel_counts = rowSums.value_counts()
    sns.set(font_scale=2)
    plt.figure(figsize=(15, 8))
    ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)
    plt.title("Images having multiple labels ")
    plt.ylabel('Number of conditions', fontsize=18)
    plt.xlabel('Number of labels', fontsize=18)

    rects = ax.patches
    labels = multiLabel_counts.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')
    plt.show()


def build_class_weights(data):
    # Reweights classes by their frequency so that the weights sum to num_classes
    epid, intra, sub = data.sum().epidural, data.sum().intraparenchymal, data.sum().subarachnoid
    total = epid + intra + sub
    epid_scalar, intra_scalar, sub_scalar = total / epid, total / intra, total / sub
    scalar_totals = epid_scalar + intra_scalar + sub_scalar
    epid_rescaled, intra_rescaled, sub_rescaled = 3 * epid_scalar / scalar_totals, 3 * intra_scalar / scalar_totals, 3 * sub_scalar / scalar_totals
    return epid_rescaled, intra_rescaled, sub_rescaled


def print_model_metrics(y_pred, y_true):
    print(f'Hamming Loss: {hamming_loss(y_pred, y_true):.2%}')

    print('\nClassification Report')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        print(classification_report(y_pred=y_pred, y_true=y_true))

    print('\nOne vs. All Confusion Matrices')
    print(multilabel_confusion_matrix(y_pred=y_pred, y_true=y_true))


def draw_cam(model_trainer_path, image_dataset, image_index, device='cuda:0'):
    """
    Helper function to chain together classes and functions to produce a GradCAM image for easy comparison
    :param model_trainer_path: Path to pickled pretrained model_trainer class object
    :param image_dataset: Image dataset to use to generate images

    :param image_index: Index in image data set to retrieve
    :param device: 'cuda:0' for gpu or 'cpu' for cpu
    :return: Displays set of 4 images, the original image along with a GradCAM image for each class
    """
    model_trainer = load_model_trainer(model_trainer_path)
    model_gradcam = ResNeXt(num_classes=3, device=device)
    model_gradcam.resnext.load_state_dict(model_trainer.model.state_dict())

    image = image_dataset[image_index]['image'].to(device)

    model_gradcam.draw_cam(image, true_label=image_dataset.labels.iloc[image_index, 1:].values if image_dataset.labels is not None else None)


def plot_auc(probs, labels, title, lw=2):
    # Compute macro-average ROC curve and ROC area
    n_classes = probs.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true=labels[:, i], y_score=probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(12, 12))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + ' Multi-Class ROC')
    plt.legend(loc="lower right")
    plt.show()


def test_threshold(probs, y_true, threshold):
    decs = probs.iloc[:, 1:].applymap(lambda x: 1 if x > threshold else 0)
    print_model_metrics(decs, y_true)
