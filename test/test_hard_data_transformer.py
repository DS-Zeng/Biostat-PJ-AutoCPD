'''
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2025-05-28
Description    : Test script comparing simple NN and Transformer on challenging multivariate data,
                with both positive and negative examples generated by the same ultra-challenging
                data generator (negative examples have zero mean but keep noise & dynamics).
'''
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_docs.modeling as tfdoc_model
import tensorflow_docs.plots as tfdoc_plot
from sklearn.utils import shuffle

from autocpd.neuralnetwork import compile_and_fit, general_simple_nn
from tensorflow.keras import layers, models

# --- Transformer definition ---
def transformer_nn(n, d_model=64, num_heads=4, ff_dim=128, num_classes=2):
    inputs = tf.keras.Input(shape=(n,))
    x = layers.Reshape((n, 1))(inputs)
    x = layers.Dense(d_model)(x)
    pos_emb = layers.Embedding(input_dim=n, output_dim=d_model)
    positions = tf.range(start=0, limit=n, delta=1)
    x += pos_emb(positions)
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)
    ffn = layers.Dense(ff_dim, activation="relu")(x)
    ffn = layers.Dense(d_model)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs)

# --- Ultra-challenging multivariate data generator ---
class DataGenUltraChallenging:
    def __init__(self, N_sub, n, d=5, random_state=None):
        self.N_sub, self.n, self.d = N_sub, n, d
        S = 3
        # segment-wise mean & variance
        self.mu_segments  = [np.zeros(d) + i       for i in range(S)]
        self.var_segments = [np.ones(d) * (i + 1)  for i in range(S)]
        # AR(2) coefs
        self.ar_coefs     = [(0.5, -0.3)] * d
        # nonlinear trends
        self.trends       = [lambda t, j=j: 0.005*(j+1)*t**(1 + 0.2*j) for j in range(S)]
        # seasonal components
        self.seasons      = [(10, 0.3), (25, 0.7)]
        self.df           = 3
        self.tau_bounds   = [(int(n*0.2), int(n*0.4)), (int(n*0.4), int(n*0.8))]
        if random_state is not None:
            np.random.seed(random_state)

    def _generate_one(self):
        taus = sorted([np.random.randint(lb, ub) for lb, ub in self.tau_bounds])
        cp = [0] + taus + [self.n]
        X = np.zeros((self.n, self.d))
        E = np.random.standard_t(self.df, size=(self.n, self.d))
        for t in range(self.n):
            # find current segment
            seg = next(i for i in range(len(cp)-1) if cp[i] <= t < cp[i+1])
            mu_t  = self.mu_segments[seg]
            var_t = self.var_segments[seg]
            trend = np.array([self.trends[seg](t) for _ in range(self.d)])
            seasonal = sum(A * np.sin(2*np.pi*t/P) for P, A in self.seasons)
            ar_term = np.zeros(self.d)
            if t > 0:
                ar_term += np.array([self.ar_coefs[j][0] * X[t-1, j] for j in range(self.d)])
            if t > 1:
                ar_term += np.array([self.ar_coefs[j][1] * X[t-2, j] for j in range(self.d)])
            X[t] = mu_t + trend + seasonal + ar_term + var_t * E[t]
        return X

    def __call__(self):
        return np.array([self._generate_one() for _ in range(self.N_sub)])


# --- Test parameters ---
n, d = 100, 5
N_all     = 400
lr, epochs, bs = 1e-3, 50, 32
num_classes = 2

# number of positives/negatives
N = N_all // 2

# --- Positive examples: change-point data ---
gen = DataGenUltraChallenging(N, n, d, random_state=2025)
data_alt = gen()

# --- Negative examples: zero-mean but retain noise & dynamics ---
null_gen = DataGenUltraChallenging(N, n, d, random_state=2025)
# force all segment means to zero and variances to one
null_gen.mu_segments  = [np.zeros(d) for _ in null_gen.mu_segments]
null_gen.var_segments = [np.ones(d)  for _ in null_gen.var_segments]
data_null = null_gen()

# --- Stack, label, flatten ---
X = np.concatenate([data_alt, data_null], axis=0)             # shape (2N, n, d)
y = np.concatenate([np.ones(N), np.zeros(N)], axis=0)         # shape (2N,)

# flatten for models
X_flat = X.reshape((2*N, n*d))
X_nn   = X_flat.reshape((2*N, n*d, 1))  # for simple NN (1-channel)
X_tf   = X_flat                         # for transformer (flat input)

# shuffle
X_nn, y = shuffle(X_nn, y, random_state=42)
X_tf, _ = shuffle(X_tf, y, random_state=42)


# --- Train simple fully-connected NN ---
histories = {}
name_nn = f"simple_nn_flat_n{n*d}"
model_nn = general_simple_nn(
    n=n*d, l=2, m=50,
    num_classes=num_classes,
    model_name=name_nn
)
histories[name_nn] = compile_and_fit(
    model_nn, X_nn, y, bs, lr,
    name_nn, Path("logs"),
    tfdoc_model.EpochDots(),
    max_epochs=epochs
)

# --- Train transformer ---
name_tf = f"transformer_flat_n{n*d}"
model_tf = transformer_nn(n*d)
histories[name_tf] = compile_and_fit(
    model_tf, X_tf, y, bs, lr,
    name_tf, Path("logs"),
    tfdoc_model.EpochDots(),
    max_epochs=epochs
)

# --- Plot accuracy comparison ---
plotter = tfdoc_plot.HistoryPlotter(metric="accuracy", smoothing_std=10)
plt.figure(figsize=(10, 6))
plotter.plot(histories)
plt.legend(loc='lower right')
plt.savefig(Path("logs", "comparison_acc.png"))

# --- Save models ---
model_nn.save(Path("logs", name_nn + ".keras"))
model_tf.save(Path("logs", name_tf + ".keras"))
