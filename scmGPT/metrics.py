from typing import Tuple
import scipy
import numpy as np

def foscttm(
        x: np.ndarray, y: np.ndarray, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")
    d = scipy.spatial.distance_matrix(x, y, **kwargs)
    foscttm_x = (d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)
    foscttm_y = (d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)
    return foscttm_x, foscttm_y

def label_transfer(ref, query, label):
    from sklearn.neighbors import KNeighborsClassifier

    X_train = ref
    y_train = label
    X_test = query

    knn = KNeighborsClassifier().fit(X_train, y_train)
    y_test = knn.predict(X_test)

    return y_test

def run_SVM(x_train, y_train, x_test, kernel="rbf", seed=2021):
    if "rbf" == kernel:
        from sklearn.svm import SVC
        model = SVC(decision_function_shape="ovr", kernel=kernel, random_state=seed)
        # model = SVC(decision_function_shape="ovo", kernel=kernel, random_state=seed)
    elif "linear" == kernel:
        from sklearn.svm import LinearSVC
        model = LinearSVC(multi_class='ovr', random_state=seed)

    ## fit model
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred