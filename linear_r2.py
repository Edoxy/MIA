import numpy as np
import pandas as pd

def generate_square(n_samples=50, random_state=None):
    rs = np.random.RandomState(random_state)
    X = rs.uniform(0., 1., (n_samples, 2))
    return X

class HyperplaneR2:
    def __init__(self, w, b):
        """
        Inizializzazione della classe.
        :param w: 1D-np.array
        :param b: scalare
        """
        self.w = w.flatten()
        self.b = b
        if np.abs(self.w[1]) > 1e-7:
            self.m = - self.w[0] / self.w[1]
            self.q = - self.b / self.w[1]
        else:
            self.m = - self.w[0] / (np.sign(self.w[1]) * 1e-7)
            self.q = - self.b / (np.sign(self.w[1]) * 1e-7)
    
    def demiplane_evaluate(self, X):
        """
        Metodo che calcola i valori dei punti (x1, x2) rispetto all'iperpiano.
        :param X: 2D-np.array di N righe e 2 colonne
        :return Y: 1D-np.array di N elementi corrispondenti al valore (w1x1 + w2x2 + b) per ognuno degli (x1,x2)
        :return y: 1D-np.array di N elementi corrispondenti alle classi +-1
        """
        Y = np.sum(X * self.w, axis=1) + self.b
        y = np.sign(Y)
        return Y, y

    def demiplane_evaluate_noise(self, X, eps=1e-1, random_state=None):
        """
        Come il metodo 'demiplane_evaluate' ma con l'aggiunta di un rumore normale sui dati (media=0, std=eps).
        """
        eps_rs = np.random.RandomState(random_state)
        eps_samples = eps_rs.normal(0., eps, X.shape[0])
        Y = np.sum(X * self.w, axis=1) + self.b + eps_samples
        y = np.sign(Y)
        return Y, y
    
    def line_x2(self, X1):
        """
        Calcola i valori X2 corrispondenti ai valori X1 in input tale che [X1,X2] sono punti appartenenti all'iperpiano.
        """
        X2 = self.m * X1 + self.q
        return X2
    
    def margin_x2(self, X1):
        """
        Calcola i valori X2 corrispondenti ai valori X1 in input tale che [X1,X2] sono punti appartenenti ai bordi del margine.
        """
        if np.abs(self.w[1]) > 1e-7:
            plus_X2 = - (self.w[0] * X1 + self.b - 1) / self.w[1]
            minus_X2 = - (self.w[0] * X1 + self.b + 1) / self.w[1]
        else:
            plus_X2 = - (self.w[0] * X1 + self.b - 1) / (np.sign(w[1]) * 1e-7)
            minus_X2 = - (self.w[0] * X1 + self.b + 1) / (np.sign(w[1]) * 1e-7)
        
        return plus_X2, minus_X2



