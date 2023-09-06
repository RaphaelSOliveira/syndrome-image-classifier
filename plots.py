# data manipulation
import numpy as np

# data visualization
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Model performance visualization
from sklearn.metrics import RocCurveDisplay

def scatterplot_2d(x:np.ndarray, y:np.ndarray, labels:np.ndarray, save_plot:bool=True) -> None:
    plot = sns.scatterplot(
        x=x,
        y=y,
        hue=labels,
        palette=sns.color_palette("hls", 10),
        legend="full",
        alpha=0.4,
    )

    plot.legend(fontsize=5)
    plt.title("2D data scatterplot using tSNE")
    plt.savefig('plots/grouped_data_2d.png') if save_plot else plt.show()

def scatterplot_3d(x:np.ndarray, y:np.ndarray, z:np.ndarray, labels:np.ndarray) -> None:    
    fig = px.scatter_3d(
        x=x,
        y=y,
        z=z,
        color=labels,
        title='Interactive 3D Scatterplot using tSNE',
        labels={'x': 'X Axis', 'y': 'Y Axis', 'z': 'Z Axis'}
    )

    fig.show()

def roc_auc_metrics_comparison_plot(fpr_cosine:np.ndarray, tpr_cosine:np.ndarray, auc_cosine:np.float64, 
                                    fpr_euclidean:np.ndarray, tpr_euclidean:np.ndarray, auc_euclidean:np.float64,
                                    save_plot:bool=True) -> None:
    
    plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr_cosine,
        tpr_cosine,
        label=f"Cosine (AUC = {auc_cosine:.2f})",
        color="deeppink",
        alpha=0.5,
        linewidth=2,
    )

    plt.plot(
        fpr_euclidean,
        tpr_euclidean,
        label=f"Euclidean (AUC = {auc_euclidean:.2f})",
        color="navy",
        alpha=0.5,
        linewidth=2,
    )

    plt.plot(
        [0, 1],
        [0, 1],
        label=f"Chance level (AUC = 0.5)",
        color='black',
        linestyle='--'
    )

    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC AUC Curve comparison between distance metrics")
    plt.legend()

    plt.savefig('plots/roc_auc_metrics_comparison_plot.png') if save_plot else plt.show()

def roc_auc_class_comparison_plot(y_onehot_test:np.ndarray, y_proba_cosine:np.ndarray, unique_labels:np.ndarray,
                                  save_plot:bool=True) -> None:
    n_classes = len(unique_labels)
    
    _, ax = plt.subplots(figsize=(6, 6))
    plt.title("ROC AUC Curve comparison between classes using cosine distance")

    colors = cm.rainbow(np.linspace(0, 1, n_classes))
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_proba_cosine[:, class_id],
            name=f"Syndrome: {unique_labels[class_id]}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 9),
        )
    
    plt.savefig('plots/roc_auc_class_comparison_plot.png') if save_plot else plt.show()