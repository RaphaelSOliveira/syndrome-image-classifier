#!/usr/bin/env python3

# event logging
import logging
logging.basicConfig(level=logging.INFO, format='(%(levelname)s) %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# data manipulation
import numpy as np

# dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# preprocessing
from sklearn.preprocessing import LabelBinarizer

# model selection
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics

# model
from sklearn.neighbors import KNeighborsClassifier

# convert list to text table
from tabulate import tabulate

# own functions
import utils
import plots
import reports

# warnings
import warnings
warnings.filterwarnings("ignore")

def main():
    logging.info("Initiating script execution...")

    logging.info("Loading dataset...")
    data = np.load('data/mini_gm_public_v0.1.p', allow_pickle=True)

    logging.info("Extracting embeddings and labels...")
    embeddings, labels = utils.extract_Xy(data)
    unique_labels = np.unique(labels)

    logging.info("Performing data dimensionality reduction and saving 2D plot...")
    pca55 = PCA(n_components=55)
    pca55_result = pca55.fit_transform(embeddings)
    print(f'Cumulative explained variation for 55 principal components: {round(np.sum(pca55.explained_variance_ratio_), 2)}')

    tsne = TSNE(method='exact', n_components=2, perplexity=50, random_state=0, verbose=1)
    tsne_results = tsne.fit_transform(pca55_result)
    tsne_2d_one = tsne_results[:,0]
    tsne_2d_two = tsne_results[:,1]

    plots.scatterplot_2d(tsne_2d_one, tsne_2d_two, labels)

    logging.info("Performing data folds, training models and saving performance metrics...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5, metric='precomputed')

    model_metrics = {
        "cosine": {"top_k": [], "roc_auc": []},
        "euclidean": {"top_k": [], "roc_auc": []}
    }

    for train_index, test_index in skf.split(embeddings, labels):
        X_train, X_test = embeddings[train_index,:], embeddings[test_index,:]
        y_train, y_test = labels[train_index], labels[test_index]

        cosine_distances_train = metrics.pairwise_distances(X_train, metric='cosine')
        cosine_distances_test = metrics.pairwise_distances(X_test, X_train, metric='cosine')

        euclidean_distances_train = metrics.pairwise_distances(X_train, metric='euclidean')
        euclidean_distances_test = metrics.pairwise_distances(X_test, X_train, metric='euclidean')

        knn.fit(cosine_distances_train, y_train)
        y_proba_cosine = knn.predict_proba(cosine_distances_test)

        knn.fit(euclidean_distances_train, y_train)
        y_proba_euclidean = knn.predict_proba(euclidean_distances_test)

        model_metrics["cosine"]["top_k"].append(metrics.top_k_accuracy_score(y_test, y_proba_cosine))
        model_metrics["cosine"]["roc_auc"].append(metrics.roc_auc_score(y_test, y_proba_cosine, average='macro', multi_class='ovr'))

        model_metrics["euclidean"]["top_k"].append(metrics.top_k_accuracy_score(y_test, y_proba_euclidean))
        model_metrics["euclidean"]["roc_auc"].append(metrics.roc_auc_score(y_test, y_proba_euclidean, average='macro', multi_class='ovr'))
   
    logging.info("Saving results into txt and pdf files...")
    metrics_table = [
        ['Distance Method', 'mean Top K', 'mean AUC'],
        ['cosine', round(np.mean(model_metrics['cosine']['top_k']), 2), round(np.mean(model_metrics['cosine']['roc_auc']), 2)],
        ['euclidean', round(np.mean(model_metrics['euclidean']['top_k']), 2), round(np.mean(model_metrics['euclidean']['roc_auc']), 2)]
    ]

    txt_metrics_table = tabulate(metrics_table, headers='firstrow', tablefmt='pretty')
    print(txt_metrics_table)

    reports.write_results_txt(txt_metrics_table)
    reports.write_results_pdf(metrics_table)

    logging.info("Calculate roc curves for both models and save comparison plots...")
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)

    fpr_cosine, tpr_cosine, _ = metrics.roc_curve(y_onehot_test.ravel(), y_proba_cosine.ravel())
    auc_cosine = metrics.auc(fpr_cosine, tpr_cosine)

    fpr_euclidean, tpr_euclidean, _ = metrics.roc_curve(y_onehot_test.ravel(), y_proba_euclidean.ravel())
    auc_euclidean = metrics.auc(fpr_euclidean, tpr_euclidean)

    plots.roc_auc_metrics_comparison_plot(fpr_cosine, tpr_cosine, auc_cosine, 
                                          fpr_euclidean, tpr_euclidean, auc_euclidean)
    
    plots.roc_auc_class_comparison_plot(y_onehot_test, y_proba_cosine, unique_labels)


    logging.info("Script executed successfully...")
if __name__ == "__main__":
    main()