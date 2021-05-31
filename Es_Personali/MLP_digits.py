import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, make_scorer
from IPython.display import display

digits_dataset = datasets.load_digits(as_frame=True)

random_state = 351871206
test_p = 0.4
val_p = 0.2

X_trainval, X_test, y_trainval, y_test = train_test_split(digits_dataset['data'].values, digits_dataset['target'].values, test_size=test_p, random_state=random_state, shuffle=True)

#display(pd.DataFrame({'X_trainval': X_trainval.shape, 'X_test': X_test.shape}, index=['N. sanmples', 'N.features']))

pca = PCA(0.95)
pca.fit(X_trainval)

#display(pd.DataFrame({'Numero PC': pca.n_components_, '% Varianza Tot. Spiegata': pca.explained_variance_ratio_.sum()}, index=['X_trainval']))

# Trasformazione dati. Salvare i vecchi in "copie di backup"

X_trainval_old = X_trainval.copy()
X_trainval = pca.transform(X_trainval)

X_test_old = X_test.copy()
X_test = pca.transform(X_test)

# Inizializzazione iper-parametri MLP
hidden_layer_sizes = 1000
activation = 'relu'
patience = 300
max_epochs = 1000
verbose = False
batch_sz = 32

# Inizializzazione MLP
mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, batch_size=batch_sz, max_iter=max_epochs, early_stopping=True, n_iter_no_change=patience, random_state=random_state ,validation_fraction=val_p)

mlp.fit(X_trainval, y_trainval)

# Performance

y_pred_trainval = mlp.predict(X_trainval)
y_pred = mlp.predict(X_test)

acc_trainval = mlp.score(X_trainval, y_trainval)
prec_trainval = precision_score(y_trainval, y_pred_trainval, average='weighted')
rec_trainval = recall_score(y_trainval, y_pred_trainval, average='weighted')
f1_trainval = f1_score(y_trainval, y_pred_trainval, average='weighted')

acc = mlp.score(X_test, y_test)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

df_perf = pd.DataFrame({'Accuracy': [acc_trainval, acc], 
                        'Precision': [prec_trainval, prec], 
                        'Recall': [rec_trainval, rec],
                        'F1': [f1_trainval, f1]}, index=['train. + val.', 'test'])

display(df_perf)
