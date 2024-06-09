import contextily as ctx
import matplotlib.pyplot as plt
import pandas as pd
import pyimpute
import rasterio

# Machine Learning
from sklearn import metrics, model_selection
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def plot_roc_curve(fper, tper, filepath):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Args:
        fper (array-like): False Positive Rate values.
        tper (array-like): True Positive Rate values.
    """
    plt.cla()
    plt.plot(fper, tper, color="red", label="ROC")
    plt.plot([0, 1], [0, 1], color="green", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend()
    plt.savefig(filepath)


def pipe_evaluate_clf(
    clf,
    X,
    y,
    name: str,
    k: int | None = None,
    test_size: float = 0.2,
    scoring: str = "f1_weighted",
    feature_names: list[str] = None,
):
    """
    Evaluate a classifier using a pipeline with data scaling and display evaluation metrics.

    Args:
        clf: The classifier to evaluate.
        X (pd.DataFrame): Features data.
        y (pd.Series): Target data.
        name (str): Name of the classifier.
        k (int, optional): Number of folds for cross-validation.
        test_size (float): Proportion of the dataset to include in the test split.
        scoring (str): Scoring metric for cross-validation.
        feature_names (list[str], optional): List of feature names.

    Returns:
        sklearn.pipeline.Pipeline: Fitted pipeline.
    """
    print(name)
    X_train, X_test, y_train, y_true = model_selection.train_test_split(
        X,
        y,
        test_size=test_size,  # Test data size
        shuffle=True,  # Shuffle the data before split
        stratify=y,  # Keeping the appearance/non-appearance ratio of Y,
        random_state=42,
    )

    if k:  # Cross-validation
        kf = model_selection.KFold(n_splits=k)  # k-fold
        scores = model_selection.cross_val_score(
            clf, X_train, y_train, cv=kf, scoring=scoring
        )
        print(
            name
            + " %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
            % (k, scores.mean() * 100, scores.std() * 200)
        )
        print()

    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)  # Classifier predictions

    # Classifier evaluation metrics
    print("Accuracy Score: %.2f" % metrics.accuracy_score(y_true, y_pred))
    print()

    print("Classification report")
    print(metrics.classification_report(y_true, y_pred))
    print()

    print("Confusion matrix")
    print(metrics.confusion_matrix(y_true, y_pred))
    print()

    print("AUC(ROC): %.2f" % metrics.roc_auc_score(y_true, y_pred))
    print()

    # ROC
    probs = pipe.predict_proba(X_test)
    prob = probs[:, 1]
    fper, tper, thresholds = metrics.roc_curve(y_true, prob)
    plot_roc_curve(fper, tper)

    if hasattr(clf, "feature_importances_"):
        print("Feature importances")
        for f, imp in zip(feature_names, clf.feature_importances_):
            print("%20s: %s" % (f, round(imp * 100, 1)))
        print()

    return pipe


def evaluate_clf(
    clf, X, y, name, k=None, test_size=0.2, scoring="f1_weighted", feature_names=None
):
    """
    Evaluate a classifier and display evaluation metrics.

    Args:
        clf: The classifier to evaluate.
        X (pd.DataFrame): Features data.
        y (pd.Series): Target data.
        name (str): Name of the classifier.
        k (int, optional): Number of folds for cross-validation.
        test_size (float): Proportion of the dataset to include in the test split.
        scoring (str): Scoring metric for cross-validation.
        feature_names (list[str], optional): List of feature names.

    Returns:
        clf: Fitted classifier.
    """
    print(name)
    X_train, X_test, y_train, y_true = model_selection.train_test_split(
        X,
        y,
        test_size=test_size,  # Test data size
        shuffle=True,  # Shuffle the data before split
        stratify=y,  # Keeping the appearance/non-appearance ratio of Y
    )

    if k:  # Cross-validation
        kf = model_selection.KFold(n_splits=k)  # k-fold
        scores = model_selection.cross_val_score(
            clf, X_train, y_train, cv=kf, scoring=scoring
        )
        print(
            name
            + " %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
            % (k, scores.mean() * 100, scores.std() * 200)
        )
        print()

    clf.fit(X_train, y_train)  # Training of classifiers
    y_pred = clf.predict(X_test)  # Classifier predictions

    # Classifier evaluation metrics
    print("Accuracy Score: %.2f" % metrics.accuracy_score(y_true, y_pred))
    print()

    print("Classification report")
    print(metrics.classification_report(y_true, y_pred))
    print()

    print("Confusion matrix")
    print(metrics.confusion_matrix(y_true, y_pred))
    print()

    print("AUC(ROC): %.2f" % metrics.roc_auc_score(y_true, y_pred))
    print()

    # ROC
    probs = clf.predict_proba(X_test)
    prob = probs[:, 1]
    fper, tper, thresholds = metrics.roc_curve(y_true, prob)
    plot_roc_curve(fper, tper)

    if hasattr(clf, "feature_importances_"):
        print("Feature importances")
        for f, imp in zip(feature_names, clf.feature_importances_):
            print("%20s: %s" % (f, round(imp * 100, 1)))
        print()

    return clf


def output_model(
    clf, tiff_output_files: list, output_dir, class_prob=True, certainty=True
):
    """
    Impute raster data using a trained classifier and save the output.

    Args:
        clf: The trained classifier.
        tiff_output_files (list): List of input TIFF files.
        output_dir (str): Directory to save the output.
        class_prob (bool): Whether to output class probabilities.
        certainty (bool): Whether to output certainty scores.

    Returns:
        None
    """
    target_xs, raster_info = pyimpute.load_targets(tiff_output_files)
    pyimpute.impute(
        target_xs,
        clf,
        raster_info,
        outdir=output_dir,
        class_prob=class_prob,
        certainty=certainty,
    )
    return


def plotit(x, title, cmap="Blues"):
    """
    Plot a 2D array with a specified colormap.

    Args:
        x (array-like): 2D array to plot.
        title (str): Title of the plot.
        cmap (str): Colormap for the plot.

    Returns:
        None
    """
    plt.figure(figsize=(14, 7))
    plt.imshow(x, cmap=cmap, interpolation="nearest")
    plt.colorbar()
    plt.title(title, fontweight="bold")
    plt.show()


def get_dist_avg(output_files: list[str]):
    """
    Calculate the average distribution from a list of raster files.

    Args:
        output_files (list[str]): List of paths to raster files.

    Returns:
        np.ndarray: Array representing the average distribution.
    """
    n_files = len(output_files)

    dist_0 = rasterio.open(output_files[0]).read(1)
    for file in output_files[1:]:
        dist = rasterio.open(file).read(1)
        dist_0 += dist

    return dist_0 / n_files
