# %% [markdown]
# # Retele neuronale + Backpropagation - explicat pas cu pas
#
# Continuam exact de unde ai ramas cu regresia liniara (acelasi set de date,
# aceeasi idee de normalizare + gradient descent). Diferenta: aici modelul
# nu mai e o simpla dreapta `y = a*x + b`, ci o mica "retea" cu un strat
# ascuns, care poate invata si o CURBA, nu doar o linie dreapta.

# %%
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# # 1. Datele
#
# Le pastram identice cu exemplul de regresie liniara.

# %%
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

suprafete = apartamente[:, 0]
preturi = apartamente[:, 1]

n = len(suprafete)
n

# %% [markdown]
# ## Este `y` (pretul) folosit la intrare?
#
# **NU.** Asta e un punct pe care merita sa-l lamurim de la inceput, pentru
# ca e sursa clasica de confuzie.
#
# - `X` = suprafata (m2) -> **singura** informatie pe care reteaua o
#   primeste ca "intrare" si o foloseste ca sa calculeze o predictie.
# - `y` = pretul real -> nu intra NICIODATA in formula de predictie.
#   Il folosim DOAR dupa ce am facut o predictie, ca sa masuram cat de
#   gresita a fost (`eroare = y_pred - y`), si din eroarea aia calculam
#   gradientii care ajusteaza ponderile.
#
# Cu alte cuvinte: `y` e "cheia de corectura", nu un ingredient al
# predictiei. Daca `y` ar intra si el in reteaua care face predictia,
# am "trisa" (reteaua ar vedea raspunsul inainte sa-l ghiceasca).

# %%
X_raw = suprafete.reshape(-1, 1).astype(float)   # shape (10, 1) -> 10 exemple, 1 caracteristica
y_raw = preturi.reshape(-1, 1).astype(float)      # shape (10, 1) -> 10 exemple, 1 valoare tinta

print("X_raw (suprafata, intrarea retelei):\n", X_raw.T)
print("y_raw (pretul, folosit DOAR pentru eroare/loss):\n", y_raw.T)
print("X_raw.shape:", X_raw.shape, " y_raw.shape:", y_raw.shape)

# %% [markdown]
# # 2. Normalizare
#
# Exact ca la regresia liniara: aducem datele la medie 0 si deviatie
# standard 1. Fara asta, gradientii pentru `suprafata` (scala ~30-90) si
# pentru `pret` (scala ~50.000-350.000) ar avea marimi complet diferite,
# iar antrenarea ar deveni instabila sau foarte lenta.

# %%
x_mean, x_std = X_raw.mean(), X_raw.std()
y_mean, y_std = y_raw.mean(), y_raw.std()

X = (X_raw - x_mean) / x_std
y = (y_raw - y_mean) / y_std

print("X normalizat:\n", X.T)
print("y normalizat:\n", y.T)

# %% [markdown]
# # 3. Se poate si fara "neuroni ascunsi"? Da - e regresia liniara!
#
# O retea fara strat ascuns e doar: `y_pred = X @ W + b`.
# Cu `X` de shape (10,1) si `W` de shape (1,1), asta e MATEMATIC identic
# cu `y_pred = a*x + b`. Hai sa o antrenam cu exact acelasi gradient
# descent ca in scriptul tau, doar scris in forma de matrice, ca sa vezi
# ca da acelasi rezultat.

# %%
W0 = np.zeros((1, 1))   # echivalentul lui "a"
b0 = np.zeros((1, 1))   # echivalentul lui "b"
alpha = 0.1
epochs_liniar = 1000

for i in range(epochs_liniar):
    y_pred0 = X @ W0 + b0                  # forward pass (liniar, fara activare)
    erori0 = y_pred0 - y
    dW0 = (2 / n) * (X.T @ erori0)         # exact grad_a de la tine, dar in forma matriceala
    db0 = (2 / n) * np.sum(erori0)         # exact grad_b de la tine
    W0 -= alpha * dW0
    b0 -= alpha * db0

# transformam inapoi in scara reala (EUR), la fel ca in scriptul tau
a_din_retea_liniara = W0[0, 0] * (y_std / x_std)
b_din_retea_liniara = y_mean + b0[0, 0] * y_std - a_din_retea_liniara * x_mean
print("a, b obtinute din 'reteaua' fara strat ascuns:", a_din_retea_liniara, b_din_retea_liniara)
print("(comparativ cu ce ai obtinut tu analitic: a=4110.22, b=-72687.58 - ar trebui sa fie apropiate)")

# %% [markdown]
# Concluzie: **fara strat ascuns, o "retea neuronala" e doar regresie
# liniara** - poate invata NUMAI o dreapta. Ca sa prindem curbura din
# date (pretul creste mai rapid pe masura ce suprafata creste), avem
# nevoie de un strat ascuns cu o activare NELINIARA.

# %% [markdown]
# # 4. Ce inseamna operatorul `@`?
#
# `@` este inmultirea de matrice. Pentru doua matrice `A` (m x k) si
# `B` (k x n), rezultatul `A @ B` are shape (m x n), iar fiecare element
# se calculeaza asa:
#
# ```
# (A @ B)[i, j] = suma peste k din A[i, k] * B[k, j]
# ```
#
# Practic: "randul i din A" inmultit, element cu element, cu "coloana j
# din B", apoi adunate. Hai sa verificam asta cu un exemplu mic si cu
# bucle `for`, ca sa vedem ca `@` nu e magie, e doar o prescurtare pentru
# niste sume pe care le-am scrie oricum cu bucle.

# %%
A_demo = np.array([[1.0], [2.0], [3.0]])              # shape (3, 1) -> 3 "exemple", 1 caracteristica
W_demo = np.array([[10.0, 20.0, 30.0, 40.0]])         # shape (1, 4) -> 1 caracteristica, 4 neuroni

produs_cu_at = A_demo @ W_demo
print("A_demo @ W_demo =\n", produs_cu_at)

produs_manual = np.zeros((3, 4))
for i in range(3):          # pentru fiecare exemplu (rand din A_demo)
    for j in range(4):      # pentru fiecare neuron (coloana din W_demo)
        suma = 0.0
        for k in range(1):  # pentru fiecare caracteristica de intrare (aici doar 1)
            suma += A_demo[i, k] * W_demo[k, j]
        produs_manual[i, j] = suma

print("Acelasi calcul, cu bucle for =\n", produs_manual)
print("Sunt identice?", np.allclose(produs_cu_at, produs_manual))

# %% [markdown]
# In reteaua noastra, `X @ W1` face exact asta: pentru fiecare apartament
# (rand din `X`) si fiecare neuron ascuns (coloana din `W1`), calculeaza
# o combinatie liniara a intrarilor. Diferenta fata de regresia liniara e
# ca acum avem 4 "coloane" (4 neuroni), deci obtinem 4 valori pentru
# fiecare apartament, nu doar una singura.

# %% [markdown]
# # 5. Arhitectura retelei
#
# ```
#   intrare (1: suprafata normalizata)
#         |
#         v   W1 (1x4), b1 (1x4)
#   strat ascuns: z1 = X @ W1 + b1        (4 valori pentru fiecare exemplu)
#         |
#         v   activare neliniara: a1 = tanh(z1)
#         |
#         v   W2 (4x1), b2 (1x1)
#   iesire: z2 = a1 @ W2 + b2  = y_pred    (1 valoare: pretul prezis)
# ```
#
# De ce avem nevoie de activarea `tanh` intre cele doua straturi? Daca am
# lega direct `z2 = (X @ W1 + b1) @ W2 + b2`, matematic asta se reduce tot
# la o formula liniara in `X` (o compunere de functii liniare e tot
# liniara!). Neliniaritatea (tanh) e cea care permite retelei sa "indoaie"
# linia si sa aproximeze o curba.

# %%
np.random.seed(42)
H = 4   # cati neuroni are stratul ascuns

W1 = np.random.randn(1, H) * 0.5
b1 = np.zeros((1, H))
W2 = np.random.randn(H, 1) * 0.5
b2 = np.zeros((1, 1))

print("W1 (1 intrare -> 4 neuroni ascunsi):\n", W1)
print("W2 (4 neuroni ascunsi -> 1 iesire):\n", W2)

# %% [markdown]
# # 6. Un singur forward pass, facut manual (inainte de bucla), ca sa vedem valorile

# %%
z1 = X @ W1 + b1
a1 = np.tanh(z1)
z2 = a1 @ W2 + b2
y_pred = z2

print("z1 (primele 2 apartamente):\n", z1[:2])
print("a1 = tanh(z1) (primele 2 apartamente):\n", a1[:2])
print("y_pred (primele 2 apartamente, normalizat):\n", y_pred[:2])

# %%
erori = y_pred - y
L = np.mean(erori ** 2)
print("Loss initial (MSE, pe date normalizate):", L)

# %% [markdown]
# # 7. Derivarea formulei de backpropagation
#
# Ideea e IDENTICA cu gradientii de la regresia liniara
# (`grad_a = (2/n) * sum(erori * x)`), doar ca acum aplicam regula lantului
# (chain rule) prin DOUA straturi in loc de unul singur.
#
# Cost (MSE):
# ```
#     L = (1/N) * suma( (y_pred_i - y_i)^2 )
# ```
#
# **Stratul de iesire** (z2 = y_pred, fara activare, e liniar):
# ```
#     dL/dz2 = (2/N) * (y_pred - y)              <- exact aceeasi formula ca la regresia liniara!
#     dL/dW2 = a1^T @ dz2                        <- pentru ca z2 = a1 @ W2 + b2, deci dz2/dW2 = a1
#     dL/db2 = suma(dz2) pe toate exemplele
# ```
#
# **Stratul ascuns** (z1 -> a1 = tanh(z1)):
# ```
#     dL/da1 = dz2 @ W2^T          <- "trimitem" eroarea inapoi, prin ponderile W2
#     dL/dz1 = dL/da1 * (1 - a1^2) <- trecem prin derivata lui tanh (regula lantului)
#     dL/dW1 = X^T @ dz1
#     dL/db1 = suma(dz1) pe toate exemplele
# ```
#
# De ce `1 - a1^2`? Pentru ca daca `a = tanh(z)`, atunci
# `d(tanh)/dz = 1 - tanh(z)^2 = 1 - a^2`. E o proprietate a functiei tanh
# care face calculul convenabil (nu trebuie sa recalculam z, folosim
# direct valoarea deja calculata `a1`).
#
# De ce apar transpuse (`.T`) si `@`? Pentru ca acum avem MAI MULTI
# neuroni si MAI MULTE exemple in acelasi timp - `.T @` e felul in care
# "adunam" contributia fiecarui exemplu la gradientul fiecarei ponderi,
# exact cum la regresia liniara foloseam `np.sum(...)` peste toate
# exemplele.

# %% [markdown]
# ## Un singur backward pass, facut manual (inainte de bucla)

# %%
dz2 = (2 / n) * erori                     # (10, 1)
dW2 = a1.T @ dz2                          # (4, 1)
db2 = np.sum(dz2, axis=0, keepdims=True)  # (1, 1)

da1 = dz2 @ W2.T                          # (10, 4) - eroarea trimisa inapoi in stratul ascuns
dz1 = da1 * (1 - a1 ** 2)                 # (10, 4) - trecuta prin derivata lui tanh
dW1 = X.T @ dz1                           # (1, 4)
db1 = np.sum(dz1, axis=0, keepdims=True)  # (1, 4)

print("dW2:\n", dW2)
print("dW1:\n", dW1)

# %% [markdown]
# ## Verificam echivalenta `@` cu bucle for, pentru dW1 (ca sa fie clar ca nu e magie)

# %%
dW1_manual = np.zeros((1, H))
for j in range(H):                      # pentru fiecare neuron ascuns
    suma = 0.0
    for i in range(n):                  # pentru fiecare exemplu (apartament)
        suma += X[i, 0] * dz1[i, j]
    dW1_manual[0, j] = suma

print("dW1 (cu @):   ", dW1)
print("dW1 (manual): ", dW1_manual)
print("Sunt identice?", np.allclose(dW1, dW1_manual))

# %% [markdown]
# # 8. De ce `tanh` si nu `ReLU`?
#
# Cateva motive concrete, relevante mai ales pe un set de date MIC ca al
# nostru (10 exemple, 4 neuroni):
#
# 1. **Continuitatea derivatei.** `tanh` are derivata `1 - tanh(z)^2`,
#    definita si neteda peste tot. `ReLU` are derivata 0 pentru z<0 si 1
#    pentru z>0 - o discontinuitate la 0.
# 2. **"Neuroni morti".** Daca un neuron ReLU ajunge sa produca mereu
#    valori negative (deci `relu(z)=0`), gradientul lui devine 0 si
#    neuronul nu mai invata NICIODATA (e "mort"). Cu doar 4 neuroni, daca
#    2-3 mor, reteaua ramane cu foarte putina capacitate. `tanh` nu are
#    aceasta problema - iesirea lui nu e niciodata "blocata" la 0.
# 3. **Centrare in jurul lui 0.** `tanh` produce valori intre -1 si 1,
#    centrate pe 0 - se potriveste natural cu datele noastre normalizate
#    (medie 0). `ReLU` produce doar valori >= 0, ceea ce poate introduce
#    o asimetrie inutila pe un set de date atat de mic.
#
# ReLU e preferat de obicei la retele MARI, cu multe straturi si multe
# date, unde vitezele de calcul si evitarea "vanishing gradient" conteaza
# mai mult decat problema neuronilor morti. Pentru exemplul nostru mic,
# `tanh` e alegerea mai stabila si mai usor de urmarit.
#
# Hai sa comparam vizual formele celor doua functii, si apoi sa
# antrenam reteaua cu fiecare, ca sa vedem diferenta in practica.

# %%
z_demo = np.linspace(-4, 4, 200)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(z_demo, np.tanh(z_demo), label="tanh(z)")
axes[0].plot(z_demo, 1 - np.tanh(z_demo) ** 2, label="derivata tanh")
axes[0].axhline(0, color="grey", linewidth=0.7)
axes[0].set_title("tanh")
axes[0].legend()
axes[0].grid(alpha=0.3)

relu_demo = np.maximum(0, z_demo)
relu_deriv_demo = (z_demo > 0).astype(float)
axes[1].plot(z_demo, relu_demo, label="relu(z)")
axes[1].plot(z_demo, relu_deriv_demo, label="derivata relu")
axes[1].axhline(0, color="grey", linewidth=0.7)
axes[1].set_title("ReLU")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("activari_tanh_vs_relu.png", dpi=130)
print("Salvat: activari_tanh_vs_relu.png")

# %% [markdown]
# # 9. Bucla completa de antrenare (functie reutilizabila)
#
# Punem tot ce am facut manual mai sus intr-o functie, ca sa o putem
# rula usor cu diferite valori pentru `H` (numar de neuroni ascunsi) si
# diferite activari (tanh / relu), exact ca in exemplul tau unde ai
# incercat mai multe valori pentru `epochs`.

# %%
def tanh_activare(z):
    return np.tanh(z)

def tanh_derivata(a):
    return 1 - a ** 2

def relu_activare(z):
    return np.maximum(0, z)

def relu_derivata(a):
    # a = relu(z); derivata e 1 acolo unde a>0, 0 in rest
    return (a > 0).astype(float)


def antreneaza_retea(X, y, H, activare, derivata_activare, alpha=0.05, epochs=3000, seed=42):
    rng = np.random.RandomState(seed)
    W1 = rng.randn(1, H) * 0.5
    b1 = np.zeros((1, H))
    W2 = rng.randn(H, 1) * 0.5
    b2 = np.zeros((1, 1))

    n_local = X.shape[0]
    istoric = []

    for _ in range(epochs):
        # forward
        z1 = X @ W1 + b1
        a1 = activare(z1)
        z2 = a1 @ W2 + b2
        y_pred = z2

        L = np.mean((y_pred - y) ** 2)
        istoric.append(L)

        # backward
        dz2 = (2 / n_local) * (y_pred - y)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * derivata_activare(a1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # actualizare (gradient descent)
        W2 -= alpha * dW2
        b2 -= alpha * db2
        W1 -= alpha * dW1
        b1 -= alpha * db1

    return dict(W1=W1, b1=b1, W2=W2, b2=b2, istoric=istoric, activare=activare)


def prezice(model, x_intrare_normalizat):
    a1 = model["activare"](x_intrare_normalizat @ model["W1"] + model["b1"])
    return a1 @ model["W2"] + model["b2"]

# %% [markdown]
# # 10. De ce 4 neuroni ascunsi? (nu 1, nu 20)
#
# Fiecare neuron ascuns e, informal, o mica "cotitura" pe care reteaua o
# poate adauga curbei finale (o combinatie de linii "indoite" de tanh).
#
# - **1 neuron** -> aproape ca nu ai de ales decat o forma foarte simpla,
#   apropiata de o linie dreapta (similar cu regresia liniara de mai sus).
# - **4 neuroni** -> suficienta flexibilitate cat sa prinda curbura din
#   cele 10 puncte, fara sa "memoreze" fiecare punct in parte.
# - **20 de neuroni**, pe doar 10 date -> risc de "overfitting": curba se
#   poate rasuci ca sa treaca cat mai aproape de fiecare punct exact, dar
#   devine nenaturala intre puncte (nu generalizeaza).
#
# Hai sa antrenam toate cele 3 variante si sa le comparam vizual.

# %%
model_h1 = antreneaza_retea(X, y, H=1, activare=tanh_activare, derivata_activare=tanh_derivata)
model_h4 = antreneaza_retea(X, y, H=4, activare=tanh_activare, derivata_activare=tanh_derivata)
model_h20 = antreneaza_retea(X, y, H=20, activare=tanh_activare, derivata_activare=tanh_derivata)

x_linie_raw = np.linspace(X_raw.min() - 5, X_raw.max() + 5, 200).reshape(-1, 1)
x_linie_norm = (x_linie_raw - x_mean) / x_std

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, model, titlu in zip(
    axes,
    [model_h1, model_h4, model_h20],
    ["H = 1 neuron (aproape liniar)", "H = 4 neuroni (echilibrat)", "H = 20 neuroni (risc de overfitting)"],
):
    y_linie = prezice(model, x_linie_norm) * y_std + y_mean
    ax.scatter(suprafete, preturi, color="tab:blue", zorder=3)
    ax.plot(x_linie_raw, y_linie, color="tab:orange", linewidth=2)
    ax.set_title(titlu)
    ax.set_xlabel("Suprafata (m2)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Pret (EUR)")

plt.tight_layout()
plt.savefig("comparatie_numar_neuroni.png", dpi=130)
print("Salvat: comparatie_numar_neuroni.png")

# %% [markdown]
# # 11. Comparatie tanh vs relu (acelasi H=4, aceiasi hiperparametri)

# %%
model_tanh = antreneaza_retea(X, y, H=4, activare=tanh_activare, derivata_activare=tanh_derivata)
model_relu = antreneaza_retea(X, y, H=4, activare=relu_activare, derivata_activare=relu_derivata)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(model_tanh["istoric"], label="tanh")
axes[0].plot(model_relu["istoric"], label="relu")
axes[0].set_title("Evolutia loss-ului: tanh vs relu")
axes[0].set_xlabel("Epoca")
axes[0].set_ylabel("MSE (normalizat)")
axes[0].legend()
axes[0].grid(alpha=0.3)

y_linie_tanh = prezice(model_tanh, x_linie_norm) * y_std + y_mean
y_linie_relu = prezice(model_relu, x_linie_norm) * y_std + y_mean
axes[1].scatter(suprafete, preturi, color="tab:blue", label="Date reale", zorder=3)
axes[1].plot(x_linie_raw, y_linie_tanh, color="tab:orange", label="tanh", linewidth=2)
axes[1].plot(x_linie_raw, y_linie_relu, color="tab:green", label="relu", linewidth=2, linestyle="--")
axes[1].set_title("Curba invatata: tanh vs relu")
axes[1].set_xlabel("Suprafata (m2)")
axes[1].set_ylabel("Pret (EUR)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("comparatie_tanh_vs_relu.png", dpi=130)
print("Salvat: comparatie_tanh_vs_relu.png")
print("Loss final tanh:", model_tanh["istoric"][-1])
print("Loss final relu:", model_relu["istoric"][-1])

# %% [markdown]
# # 12. Rezultatul final (modelul cu H=4, tanh) - predictii pe date reale

# %%
y_pred_final = prezice(model_h4, X) * y_std + y_mean

for suprafata, pret_real, pret_prezis in zip(suprafete, preturi, y_pred_final.ravel()):
    print(f"suprafata={suprafata:3d} m2   pret_real={pret_real:8.0f} EUR   pret_prezis={pret_prezis:8.0f} EUR")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(model_h4["istoric"])
axes[0].set_title("Scaderea erorii (loss) - H=4, tanh")
axes[0].set_xlabel("Epoca")
axes[0].set_ylabel("MSE (normalizat)")
axes[0].grid(alpha=0.3)

y_linie_final = prezice(model_h4, x_linie_norm) * y_std + y_mean
axes[1].scatter(suprafete, preturi, color="tab:blue", label="Date reale", zorder=3)
axes[1].plot(x_linie_raw, y_linie_final, color="tab:orange", label="Predictia retelei", linewidth=2)
axes[1].set_title("Suprafata vs. Pret - retea neuronala (H=4, tanh)")
axes[1].set_xlabel("Suprafata (m2)")
axes[1].set_ylabel("Pret (EUR)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("rezultat_final.png", dpi=130)
print("Salvat: rezultat_final.png")
