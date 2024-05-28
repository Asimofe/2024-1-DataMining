import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

class SoftMarginSVM:
    def __init__(self, C=1.0):
        self.C = C
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.astype(float)

        # Kernel matrix
        K = np.outer(y, y) * (X @ X.T)

        # Convert to cvxopt format
        P = matrix(K, tc='d')
        q = matrix(-np.ones(n_samples), tc='d')
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))), tc='d')
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)), tc='d')
        A = matrix(y, (1, n_samples), tc='d')
        b = matrix(0.0, tc='d')

        # Solve QP problem
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = alphas > 1e-5
        ind = np.arange(len(alphas))[sv]
        self.alphas = alphas[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]

        # Weight vector
        self.w = np.sum(self.alphas[:, None] * self.sv_y[:, None] * self.sv_X, axis=0)

        # Intercept term
        self.b = np.mean(self.sv_y - self.sv_X @ self.w)

    def project(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.project(X))

def preprocess_data(df):
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    df['X6'] = df['X6'].map(month_map)
    df['X11'] = df['X11'].map({'Returning_Visitor': 1, 'New_Visitor': 0})
    df['X12'] = df['X12'].map({True: 1, False: 0})
    return df

def normalize_data(df):
    # Ensure all columns are float64
    df = df.astype('float64')
    return (df - df.mean()) / df.std()

# Load and preprocess train data
train_df = pd.read_csv('data/purchase_train.csv')
print("First few rows of the train dataset:")
print(train_df.head())
train_df = preprocess_data(train_df)

# Normalize data
train_df.iloc[:, :-1] = normalize_data(train_df.iloc[:, :-1])

# Separate features and target
X_train = train_df.drop(columns=['y']).values
y_train = train_df['y'].map({True: 1, False: -1}).values  # Convert target to 1 and -1

# Load and preprocess test data
test_df = pd.read_csv('data/purchase_test.csv')
print("First few rows of the test dataset:")
print(test_df.head())
test_df = preprocess_data(test_df)

# Normalize data
test_df = normalize_data(test_df)

# Extract test features
X_test = test_df.values

# Create Soft Margin SVM classifier
svm = SoftMarginSVM(C=1.0)
svm.fit(X_train, y_train)

# Predict on test data
predictions = svm.predict(X_test)

# Save predictions to CSV
output_df = pd.DataFrame({'Prediction': predictions})
output_df.to_csv('data/predictions.csv', index=False)

print("Predictions saved to predictions.csv")
