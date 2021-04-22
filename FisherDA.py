import numpy as np


class MultipleFisherDiscriminantAnalysis:
    def __init__(self, n_dimensions=None):
        self.n_dimensions_ = n_dimensions
        self.components_ = None
        self.within_ = None
        self.between_ = None
        self.eigenvectors_ = None  # Matrice con colonne che sono gli autovettori
        self.eigenvalues_ = None  # i-esimo autovalore corrispondente all'i-esima colonna/autovettore in self.eigenvectors_
        
    def fit(self, X, y):
        """
        param X: numpy array (matrice) N-by-n di N istanze descritte da n features
        param y: numpy array (vettore) di N elementi tale che y[i] indica la classe di X[i, :]
        """
        
        N, n = X.shape
        classes = list(np.unique(y))
        if self.n_dimensions_ is None:
            self.n_dimensions_ = min((len(classes) - 1), n)
        elif self.n_dimensions_ > min((len(classes) - 1), n):
            raise ValueError("Attribute n_dimensions_ must be less than or equal to min((len(list(np.unique(y))) - 1), X.shape[1]).")
        
        # Vettore medio del dataset
        m = X.mean(axis=0)
        
        # Inizializzazione matrice "within class scatter matrix"
        self.within_ = np.zeros((n, n))
        # Inizializzazione matrice avente per righe i centroidi delle classi
        M = np.zeros((np.unique(y).size, n))
        # Inizializzazione lista contenente cardinalità classi
        Ns = np.zeros(np.unique(y).size)
        
        for i in range(len(classes)):
            Xi = X[y == y[i], :]
            Ni = Xi.shape[0]
            mi = Xi.mean(axis=0)
            
            Xi_ = Xi - mi
            
            Si = Xi_.T @ Xi_
            # EQUIVALENTEMENTE:
            # Si = (Ni - 1) * np.cov(Xi.T)
            
            M[i, :] = mi
            Ns[i] = Ni
            
            self.within_ = self.within_ + Si
        
        # Calcolo della matrice "between class scatter matrix"
        M_ = (M - m) * np.sqrt(np.expand_dims(Ns, axis=1))
        self.between_ = M_.T @ M_
        
        # Calcolo della matrice S_ := (Sw^{-1} @ Sb) di cui dobbiamo calcolare gli autovalori/vettori
        # ATTENZIONE: Evitare il più possibile il calcolo diretto dell'inversa!
        # SUGGERIMENTO: risolvere il sistema lineare Sw @ X = Sb, poiché computazionalmente più efficiente.
        
        # Assumendo Sw invertibile:
        try:
            S_ = np.linalg.solve(self.within_, self.between_)
        except np.linalg.LinAlgError:
            # Assumendo Sw NON invertibile:
            S_, _, _, _ = np.linalg.lstsq(self.within_, self.between_, rcond=None)
        
        # Calcolo autovalori e autovettori
        self.eigenvalues_ , self.eigenvectors_ = np.linalg.eig(S_)
        
        # Selezione degli n_dimensions autovalori con val. assoluti maggiori (conserviamo solo quelli ed i risp. autovettori)
        eigen_ii = np.argsort(np.abs(self.eigenvalues_))
        eigen_ii = eigen_ii[-1::-1]
        
        self.eigenvalues_ = self.eigenvalues_[eigen_ii[:self.n_dimensions_]]
        self.eigenvectors_ = self.eigenvectors_[:, eigen_ii[:self.n_dimensions_]]
        
    def transform(self, X):
        # Calcolo della proiezione dei dati in X rispetto agli autovettori calcolati col metodo fit.
        # 
        # RICORDA:
        # dato x in R^n vettore colonna e A matrice n-by-m, 
        # il vettore z rappresentante x proiettato sullo spazio delle colonne di A è dato da:
        # z = A_trasp @ x
        #
        
        # ESERCIZIO: scrivere la formula per proiettare X sullo spazio generato dagli autovettori in self.eigenvectors_
        # SUGGERIMENTO: ricordare che i vettori da proiettare sono le rige di X!
        
        Z = X @ self.eigenvectors_
        
        return Z

    
    
