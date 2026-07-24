"""
Exemplu simplu de retea neuronala + backpropagation, scrise "de mana" cu numpy.
Fara keras / pytorch / tensorflow - doar numpy, pandas, matplotlib.

Problema: pornind de la suprafata unui apartament (m2), vrem sa prezicem pretul (EUR).

Arhitectura retelei (foarte mica, ca sa fie usor de urmarit):

    input (1 neuron: suprafata)
        -> strat ascuns (4 neuroni, activare tanh)
            -> output (1 neuron, liniar: pretul prezis)

Notatii folosite mai jos (stil "manual de matematica"):
    X        - input-urile (suprafata, normalizata)
    y        - target-urile (pretul, normalizat)
    W1, b1   - ponderi/bias intre input si stratul ascuns
    W2, b2   - ponderi/bias intre stratul ascuns si output
    z1       - suma ponderata inainte de activare, in stratul ascuns
    a1       - activarea (tanh) stratului ascuns
    y_pred   - iesirea finala a retelei (z2, liniar)
    L        - functia de cost (eroarea medie patratica - MSE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Datele
# ---------------------------------------------------------------
apartamente = np.array([
    (30, 54_000),
    (36, 75_000),
    (49, 120_000),
    (53, 175_000),
    (69, 193_000),
    (70, 215_000),
    (79, 210_000),
    (83, 245_000),
    (84, 284_000),
    (90, 345_000),
])

df = pd.DataFrame(apartamente, columns=["suprafata_m2", "pret_eur"])
print(df)

# ---------------------------------------------------------------
# 2. Normalizare
# ---------------------------------------------------------------
# Retelele neuronale invata mult mai bine/rapid cand datele au
# medie 0 si deviatie standard 1 (altfel gradientii "explodeaza"
# sau sunt foarte mici din cauza scalei diferite intre m2 si EUR).
X_raw = df["suprafata_m2"].values.reshape(-1, 1).astype(float)  # (N, 1)
y_raw = df["pret_eur"].values.reshape(-1, 1).astype(float)      # (N, 1)

x_mean, x_std = X_raw.mean(), X_raw.std()
y_mean, y_std = y_raw.mean(), y_raw.std()

X = (X_raw - x_mean) / x_std
y = (y_raw - y_mean) / y_std

N = X.shape[0]   # numarul de exemple (10 apartamente)

# ---------------------------------------------------------------
# 3. Initializarea retelei
# ---------------------------------------------------------------
np.random.seed(42)  # ca sa obtinem mereu acelasi rezultat la rulari repetate

H = 4  # cati neuroni are stratul ascuns

# ponderi mici, aleatoare - daca le porneam cu 0, toti neuronii ar
# invata acelasi lucru (simetrie perfecta) si reteaua nu ar avea rost
W1 = np.random.randn(1, H) * 0.5   # (1, H)
b1 = np.zeros((1, H))              # (1, H)

W2 = np.random.randn(H, 1) * 0.5   # (H, 1)
b2 = np.zeros((1, 1))              # (1, 1)

lr = 0.05          # learning rate (pasul de invatare)
epochs = 3000       # de cate ori "vede" reteaua tot setul de date
istoric_loss = []


def tanh(z):
    return np.tanh(z)


def tanh_derivata(a):
    # daca a = tanh(z), atunci d(tanh)/dz = 1 - tanh(z)^2 = 1 - a^2
    return 1 - a ** 2


# ---------------------------------------------------------------
# 4. Bucla de antrenare: forward pass + backward pass (backprop)
# ---------------------------------------------------------------
for epoca in range(epochs):

    # ----- FORWARD PASS -----
    # trecem datele "inainte" prin retea, strat cu strat
    z1 = X @ W1 + b1        # (N, H)  suma ponderata in stratul ascuns
    a1 = tanh(z1)            # (N, H)  activare neliniara

    z2 = a1 @ W2 + b2       # (N, 1)  iesirea finala (liniara)
    y_pred = z2

    # functia de cost: eroarea medie patratica (MSE)
    L = np.mean((y_pred - y) ** 2)
    istoric_loss.append(L)

    # ----- BACKWARD PASS (BACKPROPAGATION) -----
    # Ideea: aplicam regula lantului (chain rule) de la iesire spre intrare,
    # ca sa aflam cat de mult contribuie fiecare pondere la eroare (gradientul),
    # apoi ajustam ponderile in directia OPUSA gradientului.

    # dL/dz2 -> derivata costului MSE fata de iesirea neteda z2
    dz2 = 2 * (y_pred - y) / N              # (N, 1)

    # dL/dW2 si dL/db2 -> cat de mult "vinovat" e fiecare pondere din stratul 2
    dW2 = a1.T @ dz2                        # (H, 1)
    db2 = np.sum(dz2, axis=0, keepdims=True)  # (1, 1)

    # propagam eroarea inapoi in stratul ascuns
    da1 = dz2 @ W2.T                        # (N, H)  eroarea "ajunsa" in a1
    dz1 = da1 * tanh_derivata(a1)           # aplicam derivata activarii (chain rule)

    dW1 = X.T @ dz1                         # (1, H)
    db1 = np.sum(dz1, axis=0, keepdims=True)  # (1, H)

    # ----- ACTUALIZAREA PONDERILOR (gradient descent) -----
    # ne miscam in directia opusa gradientului, cu pasul "lr"
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoca % 500 == 0 or epoca == epochs - 1:
        print(f"epoca {epoca:5d}  |  loss (MSE, normalizat) = {L:.5f}")

# ---------------------------------------------------------------
# 5. Predictii pe datele de antrenare (denormalizate, in EUR)
# ---------------------------------------------------------------
z1 = tanh(X @ W1 + b1)
y_pred_norm = z1 @ W2 + b2
y_pred_eur = y_pred_norm * y_std + y_mean

df["pret_prezis_eur"] = y_pred_eur.round(0)
print("\nRezultate finale:")
print(df)

# ---------------------------------------------------------------
# 6. Grafice
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 6a. Evolutia loss-ului in timpul antrenarii
axes[0].plot(istoric_loss)
axes[0].set_title("Scaderea erorii (loss) in timpul antrenarii")
axes[0].set_xlabel("Epoca")
axes[0].set_ylabel("MSE (pe date normalizate)")
axes[0].grid(alpha=0.3)

# 6b. Curba invatata de retea, suprapusa peste datele reale
x_linie = np.linspace(X_raw.min() - 5, X_raw.max() + 5, 200).reshape(-1, 1)
x_linie_norm = (x_linie - x_mean) / x_std
a1_linie = tanh(x_linie_norm @ W1 + b1)
y_linie_norm = a1_linie @ W2 + b2
y_linie_eur = y_linie_norm * y_std + y_mean

axes[1].scatter(df["suprafata_m2"], df["pret_eur"], color="tab:blue", label="Date reale", zorder=3)
axes[1].plot(x_linie, y_linie_eur, color="tab:orange", label="Predictia retelei", linewidth=2)
axes[1].set_title("Suprafata vs. Pret - retea neuronala")
axes[1].set_xlabel("Suprafata (m2)")
axes[1].set_ylabel("Pret (EUR)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("rezultate_retea_neuronala.png", dpi=130)
print("\nGraficele au fost salvate in 'rezultate_retea_neuronala.png'")
