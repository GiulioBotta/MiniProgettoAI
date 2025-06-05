# %% [markdown]
# # Mini Progetto Intelligenza Artificiale - Riconoscimento cifre manoscritte
# **Nome:** Giulio
# **Cognome:** Bottacin
# **Matricola:** 2042340
# **Data consegna:** 5/6/2025
# ## Obiettivo
# In questo progetto esploreremo il riconoscimento di cifre manoscritte utilizzando il dataset MNIST, implementando simulazioni per studiare come diversi fattori influenzano le prestazioni dei modelli di deep learning. Analizzeremo in particolare l'impatto degli iperparametri, la robustezza al rumore e l'effetto della quantità di dati di training.

# %% [markdown]
# ## Importazione delle librerie necessarie

# %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.neural_network import MLPClassifier
from torchvision.datasets import MNIST, FashionMNIST
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# Configurazione per riproducibilità
np.random.seed(42)
tf.random.set_seed(42)
plt.rcParams['figure.figsize'] = (12, 6)

# %% [markdown]
# ## Funzioni Helper Globali

# %%
def stampa_header_esperimento(num_esp, totale, tipo_modello, config):
    """Stampa header standardizzato per esperimenti"""
    print(f"\n[{num_esp:2d}/{totale}] {tipo_modello}: {config}")
    print("-" * 50)

def stampa_risultati_esperimento(risultati):
    """Stampa risultati standardizzati per esperimenti"""
    print(f"Accuracy Training: {risultati['train_accuracy']:.4f} | Accuracy Test: {risultati['test_accuracy']:.4f}")
    print(f"Tempo: {risultati['training_time']:6.1f}s | Iterazioni: {risultati['iterations']:3d}")
    print(f"Overfitting: {risultati['overfitting']:+.4f}")

def crea_modello_cnn(tipo_architettura, learning_rate):
    """Crea modello CNN con architettura specificata"""
    model = keras.Sequential()
    
    if tipo_architettura == 'baseline':
        model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(50, activation='relu'))
    elif tipo_architettura == 'extended':
        model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
        model.add(keras.layers.MaxPooling2D(2,2))
        model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation='relu'))
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def crea_mlp_ottimale():
    """Crea MLP con configurazione ottimale identificata"""
    return MLPClassifier(
        hidden_layer_sizes=(250,),
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        tol=0.001,
        n_iter_no_change=10,
        random_state=42
    )

def crea_cnn_ottimale():
    """Crea CNN con configurazione ottimale identificata"""
    return crea_modello_cnn('extended', 0.001)

def add_gaussian_noise(images, noise_std):
    """Aggiunge rumore Gaussiano alle immagini"""
    np.random.seed(42)
    noise = np.random.normal(0, noise_std, images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0, 1)

# Variabili globali per configurazione ottimale
BEST_MLP_CONFIG = None
MLP_OPTIMAL = None

# %% [markdown]
# ## Caricamento e preparazione del dataset MNIST

# %%
# Caricamento dataset MNIST
print("Caricamento dataset MNIST...")
mnist_tr = MNIST(root="./data", train=True, download=True)
mnist_te = MNIST(root="./data", train=False, download=True)

# Conversione in array numpy
mnist_tr_data, mnist_tr_labels = mnist_tr.data.numpy(), mnist_tr.targets.numpy()
mnist_te_data, mnist_te_labels = mnist_te.data.numpy(), mnist_te.targets.numpy()

# Preprocessing per MLP (vettorizzazione e normalizzazione)
x_tr = mnist_tr_data.reshape(60000, 28 * 28) / 255.0
x_te = mnist_te_data.reshape(10000, 28 * 28) / 255.0

# Preprocessing per CNN (mantenendo formato 2D)
x_tr_conv = x_tr.reshape(-1, 28, 28, 1)
x_te_conv = x_te.reshape(-1, 28, 28, 1)

print(f"Dataset caricato: {x_tr.shape[0]} esempi di training, {x_te.shape[0]} esempi di test")

# %% [markdown]
# ## Punto A: Effetto degli iperparametri sulle prestazioni
# Analizziamo sistematicamente come variano le prestazioni dei modelli MLP e CNN al variare degli iperparametri chiave. Confronteremo 18 configurazioni MLP e 6 configurazioni CNN per un totale di 24 esperimenti mirati.
# Confronteremo 18 configurazioni MLP e 6 configurazioni CNN per un totale di 24 esperimenti mirati.

# %% [markdown]
# ### Configurazione esperimenti sistematici
# ***MLP (18 esperimenti):***
# - **Neuroni per strato**: *50, 100, 250* per testare la copertura da reti piccole a medio-grandi
# - **Numero layers**: *1 vs 2* strati nascosti per fare il confronto profondità vs larghezza
# - **Learning rate**: *0.001, 0.01, 0.1*
# ***CNN (6 esperimenti):***
# - **Filtri**: *32*, standard per MNIST, computazionalmente efficiente
# - **Architettura**: *baseline vs extended* per fare il confronto sulla complessità
# - **Learning rate**: *0.001, 0.01, 0.1*
# Per entrambi i modelli si è scelto di utilizzare il solver **Adam**, ormai standard e più performante di SDG.
# Si è volutamente scelto di eseguire meno esperimenti sulle CNN in quanto richiedono tempi molto più lunghi di training rispetto alle MLP.
# #### Scelta dei parametri di training
# ***MLP:***
# - *max_iter = 100* è sufficiente per convergenza su MNIST basato su cifre manoscritte.
# - *early_stopping = True*, previene l'overfitting essenziale quando sono presenti molti parametri.
# - *validation_fraction = 0.1*, split standard 90/10.
# - *tol = 0.001* è una precisione ragionevole per classificazione.
# - *n_iter_no_change = 10* è un livello di pazienza adeguata per permettere oscillazioni temporanee.
# ***CNN:***
# - *epochs = 20* valore di compromesso per bilanciare velocità e convergenza, il valore è più basso delle MLP perchè le CNN tipicamente convergono più velocemente.
# - *batch_size = 128*, trade-off memoria/velocità ottimale per dataset size.
# - *validation_split = 0.1*, coerente con le scelte di MLP.
# - *patience = 5*, le CNN sono meno soggette a oscillazioni quindi è stato scelto un livello di pazienza minore.
# - *min_delta = 0.001*, scelta la stessa precisione degli MLP per comparabilità diretta.
# Questa configurazione permette un confronto sistematico e bilanciato tra i due tipi di architetture.

# %% [markdown]
# #### Funzioni helper per stampe risultati

# %%
def stampa_header_esperimento(num_esp, totale, tipo_modello, config):
    print(f"\n[{num_esp:2d}/{totale}] {tipo_modello}: {config}")
    print("-" * 50)

def stampa_risultati_esperimento(risultati):
    print(f"Accuracy Training: {risultati['train_accuracy']:.4f} | Accuracy Test: {risultati['test_accuracy']:.4f}")
    print(f"Tempo: {risultati['training_time']:6.1f}s | Iterazioni: {risultati['iterations']:3d}")
    print(f"Overfitting: {risultati['overfitting']:+.4f}")

# %% [markdown]
# ### Esperimenti sistematici MLP e CNN

# %%
# Configurazione esperimenti
neuroni_lista = [50, 100, 250]
strati_lista = [1, 2]
learning_rates = [0.001, 0.01, 0.1]
architetture_cnn = ['baseline', 'extended']

risultati_mlp = []
risultati_cnn = []

print("INIZIO ESPERIMENTI MLP")
print("=" * 60)

# Esperimenti MLP
contatore = 0
esperimenti_totali = len(neuroni_lista) * len(strati_lista) * len(learning_rates)

for neuroni in neuroni_lista:
    for n_strati in strati_lista:
        for lr in learning_rates:
            contatore += 1
            
            if n_strati == 1:
                strati_nascosti = (neuroni,)
                nome_config = f"{neuroni}n_1S_lr{lr}"
            else:
                strati_nascosti = (neuroni, neuroni)
                nome_config = f"{neuroni}n_2S_lr{lr}"
            
            stampa_header_esperimento(contatore, esperimenti_totali, "MLP", nome_config)
            
            mlp = MLPClassifier(
                hidden_layer_sizes=strati_nascosti,
                learning_rate_init=lr,
                max_iter=100,
                early_stopping=True,
                validation_fraction=0.1,
                tol=0.001,
                n_iter_no_change=10,
                random_state=42
            )
            
            tempo_inizio = time.time()
            mlp.fit(x_tr, mnist_tr_labels)
            tempo_training = time.time() - tempo_inizio
            
            acc_train = mlp.score(x_tr, mnist_tr_labels)
            acc_test = mlp.score(x_te, mnist_te_labels)
            
            risultati = {
                'tipo_modello': 'MLP',
                'nome_config': nome_config,
                'neuroni': neuroni,
                'n_strati': n_strati,
                'learning_rate': lr,
                'strati_nascosti': strati_nascosti,
                'train_accuracy': acc_train,
                'test_accuracy': acc_test,
                'overfitting': acc_train - acc_test,
                'training_time': tempo_training,
                'iterations': mlp.n_iter_,
                'loss_curve': mlp.loss_curve_ if hasattr(mlp, 'loss_curve_') else [],
                'parametri_totali': sum([layer.size for layer in mlp.coefs_]) + sum([layer.size for layer in mlp.intercepts_])
            }
            
            risultati_mlp.append(risultati)
            stampa_risultati_esperimento(risultati)

print(f"\n\nINIZIO ESPERIMENTI CNN")
print("=" * 60)

# Esperimenti CNN
contatore_cnn = 0
esperimenti_totali_cnn = len(architetture_cnn) * len(learning_rates)

for arch in architetture_cnn:
    for lr in learning_rates:
        contatore_cnn += 1
        nome_config = f"CNN_{arch}_lr{lr}"
        
        stampa_header_esperimento(contatore_cnn, esperimenti_totali_cnn, "CNN", nome_config)
        
        model = crea_modello_cnn(arch, lr)
        early_stopping = keras.callbacks.EarlyStopping(
            patience=5, min_delta=0.001, restore_best_weights=True, verbose=0
        )
        
        tempo_inizio = time.time()
        history = model.fit(x_tr_conv, mnist_tr_labels, validation_split=0.1, epochs=20,
                           batch_size=128, callbacks=[early_stopping], verbose=0)
        tempo_training = time.time() - tempo_inizio
        
        train_loss, acc_train = model.evaluate(x_tr_conv, mnist_tr_labels, verbose=0)
        test_loss, acc_test = model.evaluate(x_te_conv, mnist_te_labels, verbose=0)
        
        risultati = {
            'tipo_modello': 'CNN',
            'nome_config': nome_config,
            'architettura': arch,
            'learning_rate': lr,
            'train_accuracy': acc_train,
            'test_accuracy': acc_test,
            'overfitting': acc_train - acc_test,
            'training_time': tempo_training,
            'iterations': len(history.history['loss']),
            'parametri_totali': model.count_params()
        }
        
        risultati_cnn.append(risultati)
        stampa_risultati_esperimento(risultati)

# Identificazione configurazione ottimale
migliore_mlp = max(risultati_mlp, key=lambda x: x['test_accuracy'])
migliore_cnn = max(risultati_cnn, key=lambda x: x['test_accuracy'])

BEST_MLP_CONFIG = migliore_mlp
print(f"\nCONFIGURAZIONE MLP OTTIMALE IDENTIFICATA: {migliore_mlp['nome_config']}")
print(f"Accuratezza: {migliore_mlp['test_accuracy']:.4f}")

# %% [markdown]
# ### Grafico 1: Effetto del Learning Rate sulle prestazioni MLP

# %%
# Analisi learning rate
dati_lr_001 = [r for r in risultati_mlp if r['learning_rate'] == 0.001]
dati_lr_01 = [r for r in risultati_mlp if r['learning_rate'] == 0.01]
dati_lr_1 = [r for r in risultati_mlp if r['learning_rate'] == 0.1]

acc_lr_001 = np.mean([r['test_accuracy'] for r in dati_lr_001])
acc_lr_01 = np.mean([r['test_accuracy'] for r in dati_lr_01])
acc_lr_1 = np.mean([r['test_accuracy'] for r in dati_lr_1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Curve di convergenza
for i, (dati_lr, colore, etichetta) in enumerate([(dati_lr_001, 'green', 'LR=0.001'), 
                                                   (dati_lr_01, 'blue', 'LR=0.01'), 
                                                   (dati_lr_1, 'red', 'LR=0.1')]):
    if dati_lr and dati_lr[0]['loss_curve']:
        curva_loss = dati_lr[0]['loss_curve']
        ax1.plot(range(len(curva_loss)), curva_loss, color=colore, linewidth=2, label=etichetta)

ax1.set_xlabel('Iterazioni')
ax1.set_ylabel('Loss')
ax1.set_title('Pattern di Convergenza per Learning Rate')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Accuratezza finale
learning_rates_plot = [0.001, 0.01, 0.1]
accuratezze = [acc_lr_001, acc_lr_01, acc_lr_1]
colori = ['green', 'blue', 'red']

bars = ax2.bar(range(len(learning_rates_plot)), accuratezze, color=colori, alpha=0.7)
ax2.set_xlabel('Learning Rate')
ax2.set_ylabel('Accuratezza Test Media')
ax2.set_title('Accuratezza Test per Learning Rate')
ax2.set_xticks(range(len(learning_rates_plot)))
ax2.set_xticklabels(['0.001', '0.01', '0.1'])
ax2.grid(True, alpha=0.3)

for bar, acc in zip(bars, accuratezze):
    height = bar.get_height()
    ax2.annotate(f'{acc:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Grafico 2: Confronto Completo delle Architetture

# %%
tutti_risultati = risultati_mlp + risultati_cnn
nomi_config = [r['nome_config'] for r in tutti_risultati]
acc_train_tutte = [r['train_accuracy'] for r in tutti_risultati]
acc_test_tutte = [r['test_accuracy'] for r in tutti_risultati]
tipi_modello = [r['tipo_modello'] for r in tutti_risultati]

fig, ax = plt.subplots(figsize=(16, 8))

x = np.arange(len(nomi_config))
larghezza = 0.35

bars_train = ax.bar(x - larghezza/2, acc_train_tutte, larghezza, 
                   label='Accuratezza Training', alpha=0.8, color='lightcoral')
bars_test = ax.bar(x + larghezza/2, acc_test_tutte, larghezza, 
                  label='Accuratezza Test', alpha=0.8, color='steelblue')

# Colorazione bordi diversa per MLP/CNN
for i, tipo in enumerate(tipi_modello):
    if tipo == 'MLP':
        bars_train[i].set_edgecolor('darkred')
        bars_test[i].set_edgecolor('darkblue')
        bars_train[i].set_linewidth(1.5)
        bars_test[i].set_linewidth(1.5)
    else:
        bars_train[i].set_edgecolor('orange')
        bars_test[i].set_edgecolor('green')
        bars_train[i].set_linewidth(2)
        bars_test[i].set_linewidth(2)

ax.set_xlabel('Configurazione')
ax.set_ylabel('Accuratezza')
ax.set_title('Confronto Completo: Accuratezza Training vs Test (24 Configurazioni)')
ax.set_xticks(x)
ax.set_xticklabels(nomi_config, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

# Evidenziazione migliori configurazioni
idx_migliore_mlp = tutti_risultati.index(migliore_mlp)
idx_migliore_cnn = tutti_risultati.index(migliore_cnn)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Grafico 3: Effetto Scaling MLP (1 vs 2 Strati Nascosti)

# %%
# Analisi scaling MLP
range_neuroni = neuroni_lista
acc_1_strato = []
acc_2_strati = []
tempo_1_strato = []
tempo_2_strati = []

for neuroni in range_neuroni:
    risultati_1s = [r for r in risultati_mlp if r['neuroni'] == neuroni and r['n_strati'] == 1]
    risultati_2s = [r for r in risultati_mlp if r['neuroni'] == neuroni and r['n_strati'] == 2]
    
    if risultati_1s:
        acc_1_strato.append(np.mean([r['test_accuracy'] for r in risultati_1s]))
        tempo_1_strato.append(np.mean([r['training_time'] for r in risultati_1s]))
    
    if risultati_2s:
        acc_2_strati.append(np.mean([r['test_accuracy'] for r in risultati_2s]))
        tempo_2_strati.append(np.mean([r['training_time'] for r in risultati_2s]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Accuratezza
ax1.plot(range_neuroni, acc_1_strato, 'o-', linewidth=2, markersize=8, 
         label='1 Strato Nascosto', color='blue')
ax1.plot(range_neuroni, acc_2_strati, 's-', linewidth=2, markersize=8, 
         label='2 Strati Nascosti', color='darkblue')

ax1.set_xlabel('Neuroni per Strato')
ax1.set_ylabel('Accuratezza Test')
ax1.set_title('Scaling MLP: Accuratezza vs Profondità')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Tempo di training
ax2.plot(range_neuroni, tempo_1_strato, 'o-', linewidth=2, markersize=8, 
         label='1 Strato Nascosto', color='green')
ax2.plot(range_neuroni, tempo_2_strati, 's-', linewidth=2, markersize=8, 
         label='2 Strati Nascosti', color='darkgreen')

ax2.set_xlabel('Neuroni per Strato')
ax2.set_ylabel('Tempo di Training (secondi)')
ax2.set_title('Scaling MLP: Tempo di Training vs Profondità')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Analisi quantitative aggiuntive e stampe risultati

# %%
# Calcolo metriche di efficienza
efficienze = [r['test_accuracy'] / r['training_time'] for r in tutti_risultati]
efficienza_media_mlp = np.mean([efficienze[i] for i, t in enumerate(tipi_modello) if t == 'MLP'])
efficienza_media_cnn = np.mean([efficienze[i] for i, t in enumerate(tipi_modello) if t == 'CNN'])

print("ANALISI EFFICIENZA (ACC/TEMPO):")
print("-" * 40)
print(f"Efficienza media MLP: {efficienza_media_mlp:.4f} acc/s")
print(f"Efficienza media CNN: {efficienza_media_cnn:.4f} acc/s")
print(f"Rapporto MLP/CNN: {efficienza_media_mlp/efficienza_media_cnn:.1f}x")

# Top 5 configurazioni più efficienti
top_efficienti = sorted(range(len(efficienze)), key=lambda i: efficienze[i], reverse=True)[:5]
print(f"\nTop 5 configurazioni più efficienti:")
for i, idx in enumerate(top_efficienti):
    print(f"{i+1}. {nomi_config[idx]}: {efficienze[idx]:.4f} acc/s")

# Analisi overfitting vs complessità
print(f"\nANALISI OVERFITTING VS COMPLESSITÀ:")
print("-" * 40)
complessita = [r['parametri_totali'] for r in tutti_risultati]
overfitting_vals = [r['overfitting'] for r in tutti_risultati]

print(f"Range parametri: {min(complessita)/1000:.0f}K - {max(complessita)/1000:.0f}K")
print(f"Overfitting medio MLP: {np.mean([r['overfitting'] for r in risultati_mlp]):.4f}")
print(f"Overfitting medio CNN: {np.mean([r['overfitting'] for r in risultati_cnn]):.4f}")

# Correlazione complessità-overfitting
correlazione = np.corrcoef(complessita, overfitting_vals)[0,1]
print(f"Correlazione parametri-overfitting: {correlazione:.3f}")

# Analisi velocità convergenza
print(f"\nANALISI VELOCITÀ CONVERGENZA:")
print("-" * 40)
iter_mlp = [r['iterations'] for r in risultati_mlp]
iter_cnn = [r['iterations'] for r in risultati_cnn]

print(f"Iterazioni medie MLP: {np.mean(iter_mlp):.1f}")
print(f"Iterazioni medie CNN: {np.mean(iter_cnn):.1f}")
print(f"Rapporto convergenza MLP/CNN: {np.mean(iter_mlp)/np.mean(iter_cnn):.1f}x")

# %% [markdown]
# ### Discussione finale e conclusioni Punto A
# **Architetture ottimali identificate:**
# Gli esperimenti sistematici su 24 configurazioni hanno identificato **MLP(250n_1S_lr0.001)** con **98.10%** di accuratezza come architettura leader per il riconoscimento di cifre manoscritte su MNIST. La **CNN extended con lr=0.001** raggiunge **98.85%** stabilendo il benchmark prestazionale massimo ma con costi computazionali significativamente superiori.
# **Insights critici sul Learning Rate:**
# Il learning rate emerge come iperparametro decisivo con un range ottimale di **0.001-0.01**. Nello specifico, 0.001 maximizza l'accuratezza (**97.60%** media MLP) mentre 0.01 offre il miglior compromesso velocità-prestazioni (**97.40%** media). Learning rate 0.1 causa collasso prestazionale catastrofico (**86.10%** per MLP), evidenziando l'importanza critica della calibrazione fine.
# **Profondità vs Larghezza negli MLP:**
# Controintuitivamente, le architetture a **1 strato superano sistematicamente quelle a 2 strati** con vantaggio medio di +2.2 punti percentuali, indicando che su MNIST la maggiore profondità introduce overfitting piuttosto che benefici. Questo suggerisce che la complessità intrinseca del task di riconoscimento cifre non giustifica architetture profonde, confermando che semplicity pays off per problemi ben definiti.
# **Efficienza computazionale dominata dagli MLP:**
# Gli MLP dominano l'efficienza con **12.4x** rapporto favorevole rispetto alle CNN (0.110 vs 0.009 acc/s), principalmente per tempi di training drammaticamente inferiori che compensano il gap di accuratezza. Le configurazioni MLP piccole (50-100 neuroni, LR=0.01) emergono ideali per prototipazione rapida raggiungendo >97% accuratezza in <10 secondi.
# **Controllo dell'overfitting e correlazioni:**
# Le CNN mostrano controllo superiore dell'overfitting (overfitting medio 0.006) rispetto agli MLP (0.013) grazie ai meccanismi intrinseci di regolarizzazione. La correlazione parametri-overfitting è debolmente negativa (-0.37), evidenziando che l'architettura e la regolarizzazione contano più della complessità assoluta nel controllare la generalizzazione.
# **Raccomandazioni strategiche per deployment:**
# - **Per deployment critico**: MLP(250, lr=0.001) bilancia 98.1% accuratezza con efficienza 12x superiore alle CNN
# - **Per prototipazione veloce**: MLP(100, lr=0.01) offre 97.3% accuratezza in <10 secondi con efficienza 0.22 acc/s
# - **Per massimizzazione prestazioni**: CNN extended con lr=0.001 solo quando il costo computazionale è giustificabile dal marginal gain di 0.75 punti percentuali

# %% [markdown]
# ## Punto B: Analisi delle cifre più difficili da riconoscere
# Utilizziamo l'architettura MLP ottimale identificata nel Punto A per analizzare sistematicamente quali cifre sono più difficili da classificare attraverso la matrice di confusione e l'analisi degli errori.

# %%
# Training modello ottimale per analisi errori
print("TRAINING MODELLO MLP OTTIMALE PER ANALISI ERRORI")
print("=" * 60)
print(f"Configurazione: {BEST_MLP_CONFIG['nome_config']}")
print(f"Architettura: {BEST_MLP_CONFIG['strati_nascosti']}")

MLP_OPTIMAL = crea_mlp_ottimale()
start_time = time.time()
MLP_OPTIMAL.fit(x_tr, mnist_tr_labels)
training_time = time.time() - start_time

train_accuracy = MLP_OPTIMAL.score(x_tr, mnist_tr_labels)
test_accuracy = MLP_OPTIMAL.score(x_te, mnist_te_labels)

print(f"Training completato in {training_time:.1f}s")
print(f"Accuratezza training: {train_accuracy:.4f}")
print(f"Accuratezza test: {test_accuracy:.4f}")

# Calcolo predizioni per analisi errori
y_pred = MLP_OPTIMAL.predict(x_te)
y_pred_proba = MLP_OPTIMAL.predict_proba(x_te)
total_errors = np.sum(y_pred != mnist_te_labels)

print(f"Errori totali: {total_errors}")

# %% [markdown]
# ### Grafico 1: Matrice di Confusione

# %%
cm = metrics.confusion_matrix(mnist_te_labels, y_pred)
cm_normalized = metrics.confusion_matrix(mnist_te_labels, y_pred, normalize='true')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Matrice assoluta
im1 = ax1.imshow(cm, cmap='Blues')
ax1.set_xticks(range(10))
ax1.set_yticks(range(10))
ax1.set_xlabel('Cifra Predetta', fontsize=12)
ax1.set_ylabel('Cifra Vera', fontsize=12)
ax1.set_title('Matrice di Confusione - Valori Assoluti', fontsize=14)

for i in range(10):
    for j in range(10):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax1.text(j, i, f'{cm[i, j]}', ha='center', va='center', 
                color=color, fontweight='bold')

# Matrice normalizzata
im2 = ax2.imshow(cm_normalized, cmap='Reds')
ax2.set_xticks(range(10))
ax2.set_yticks(range(10))
ax2.set_xlabel('Cifra Predetta', fontsize=12)
ax2.set_ylabel('Cifra Vera', fontsize=12)
ax2.set_title('Matrice di Confusione - Percentuali per Classe', fontsize=14)

for i in range(10):
    for j in range(10):
        color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
        ax2.text(j, i, f'{cm_normalized[i, j]:.2f}', ha='center', va='center',
                color=color, fontweight='bold')

fig.colorbar(im1, ax=ax1, shrink=0.6)
fig.colorbar(im2, ax=ax2, shrink=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Grafico 2: Difficoltà di Riconoscimento per Cifra

# %%
# Analisi errori per singola cifra
errors_per_digit = []
for digit in range(10):
    mask = mnist_te_labels == digit
    total_samples = np.sum(mask)
    correct_predictions = np.sum((y_pred == mnist_te_labels) & mask)
    errors = total_samples - correct_predictions
    error_rate = errors / total_samples
    accuracy = correct_predictions / total_samples
    
    digit_predictions = y_pred_proba[mask]
    correct_mask = (y_pred == mnist_te_labels)[mask]
    
    avg_confidence_correct = np.mean(np.max(digit_predictions[correct_mask], axis=1)) if np.any(correct_mask) else 0
    avg_confidence_errors = np.mean(np.max(digit_predictions[~correct_mask], axis=1)) if np.any(~correct_mask) else 0
    
    errors_per_digit.append({
        'digit': digit,
        'total_samples': total_samples,
        'correct': correct_predictions,
        'errors': errors,
        'error_rate': error_rate,
        'accuracy': accuracy,
        'avg_confidence_correct': avg_confidence_correct,
        'avg_confidence_errors': avg_confidence_errors
    })

df_errors = pd.DataFrame(errors_per_digit)
df_errors_sorted = df_errors.sort_values('error_rate', ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.RdYlBu_r(df_errors_sorted['error_rate'] / df_errors_sorted['error_rate'].max())
bars = ax.bar(range(10), df_errors_sorted['error_rate'] * 100, color=colors, alpha=0.8)

ax.set_xlabel('Cifra (ordinata per difficoltà)', fontsize=12)
ax.set_ylabel('Tasso di Errore (%)', fontsize=12)
ax.set_title('Difficoltà di Riconoscimento per Cifra', fontsize=14)
ax.set_xticks(range(10))
ax.set_xticklabels(df_errors_sorted['digit'])
ax.grid(True, alpha=0.3)

# Annotazioni dettagliate
for i, (bar, row) in enumerate(zip(bars, df_errors_sorted.itertuples())):
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%\n({row.errors}/{row.total_samples})', 
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Analisi quantitative aggiuntive

# %%
# Analisi Top confusioni (precedente grafico rimosso)
print("ANALISI TOP CONFUSIONI:")
print("-" * 30)

confusion_pairs = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append({
                'true_digit': i,
                'predicted_digit': j,
                'count': cm[i, j],
                'percentage_of_true': cm[i, j] / np.sum(cm[i, :]) * 100
            })

df_confusions = pd.DataFrame(confusion_pairs)
top_3_confusions = df_confusions.nlargest(3, 'count')

print("Top 3 confusioni più frequenti:")
for idx, row in top_3_confusions.iterrows():
    print(f"{row['true_digit']} → {row['predicted_digit']}: {row['count']} errori ({row['percentage_of_true']:.1f}%)")

# Analisi simmetria confusioni
print(f"\nAnalisi simmetria confusioni:")
for _, row in top_3_confusions.iterrows():
    true_digit = int(row['true_digit'])
    pred_digit = int(row['predicted_digit'])
    forward = cm[true_digit, pred_digit]
    reverse = cm[pred_digit, true_digit]
    symmetry = min(forward, reverse) / max(forward, reverse)
    print(f"Confusione {true_digit}↔{pred_digit}: Simmetria {symmetry:.2f} ({forward} vs {reverse})")

# Analisi confidenza modello (precedente subplot rimosso)
print(f"\nANALISI CONFIDENZA MODELLO:")
print("-" * 30)
print("Cifra | Conf_Corrette | Conf_Errate | Gap")
print("-" * 40)

for _, row in df_errors_sorted.iterrows():
    gap_confidenza = row['avg_confidence_correct'] - row['avg_confidence_errors']
    print(f"  {int(row['digit'])}   |     {row['avg_confidence_correct']:.3f}     |    {row['avg_confidence_errors']:.3f}    | {gap_confidenza:+.3f}")

# Correlazione confidenza-accuratezza
confidenze_corrette = df_errors_sorted['avg_confidence_correct'].values
accuratezze = df_errors_sorted['accuracy'].values
correlazione_conf = np.corrcoef(confidenze_corrette, accuratezze)[0,1]
print(f"\nCorrelazione confidenza-accuratezza: {correlazione_conf:.3f}")

# %% [markdown]
# ### Discussione finale e conclusioni Punto B
# **Gerarchia di difficoltà chiaramente stratificata:**
# L'analisi quantitativa rivela una distribuzione netta delle cifre per difficoltà con la cifra **8** che emerge come più problematica (**2.8% errori**, 28/983 campioni), seguita da **2** (**2.5%**, 26/1032) e **5** (**2.4%**, 21/892), mentre **0** e **1** si confermano più robuste (**<1% errori** ciascuna). La cifra 8 presenta due loop chiusi che creano ambiguità strutturali con 3, 6 o 9, confermando che la complessità morfologica correla direttamente con la difficoltà di classificazione.
# **Pattern di confusione morfologicamente giustificati:**
# Le Top 3 confusioni (**4→9**: 9 errori, **7→2**: 8 errori, **8→3**: 7 errori) rivelano errori che seguono logiche di similitudine visiva genuine. L'analisi della simmetria mostra pattern direzionali significativi: la confusione 4↔9 presenta simmetria moderata (0.56), mentre 8→3 mostra forte asimmetria (0.29), indicando una vulnerabilità specifica del modello nell'interpretare i loop della cifra 8 quando degradati o ambigui.
# **Meccanismo di calibrazione della confidenza eccellente:**
# Il modello dimostra autoconsapevolezza superiore con confidenze elevate per predizioni corrette (**0.990-0.998**) e significativamente ridotte per errori (**0.722-0.845**). La correlazione confidenza-accuratezza (**r=0.774**) fornisce un meccanismo naturale di quality control: soglie **<0.80** potrebbero attivare controlli manuali, mentre **>0.95** garantiscono affidabilità del 99%+.
# **Distribuzione degli errori altamente concentrata:**
# Con solo **190 errori su 10.000 esempi** (1.90% globale), il modello mostra robustezza eccellente. Le Top 3 confusioni rappresentano appena **24 errori** (12.6% del totale), indicando che non esistono vulnerabilità localizzate critiche. La distribuzione uniforme degli errori residui suggerisce che si tratta di casi di genuina ambiguità morfologica dove anche osservatori umani esperti potrebbero esitare.
# **Limiti architettonali evidenziati:**
# I risultati suggeriscono che ulteriori guadagni oltre il 98.1% richiederanno interventi sofisticati: data augmentation mirata per cifre problematiche (8, 2, 5), ensemble methods per confusioni specifiche, o architetture convoluzionali per catturare invarianze spaziali più robuste. Gli errori residui rappresentano probabilmente il limite naturale per MLP su questo livello di complessità, richiedendo approcci più sofisticati per miglioramenti marginali.

# %% [markdown]
# ## Punto C: Curve psicometriche - Effetto del rumore
# Analizziamo sistematicamente come l'accuratezza di riconoscimento degrada all'aumentare del rumore Gaussiano aggiunto alle immagini di test, utilizzando l'architettura MLP ottimale per valutare la robustezza intrinseca del modello.

# %%
# Configurazione esperimento robustezza
noise_levels = np.arange(0.00, 0.50, 0.05)
subset_size = 2000

# Campionamento stratificato
indices_stratificati = []
for digit in range(10):
    digit_indices = np.where(mnist_te_labels == digit)[0]
    n_samples = subset_size // 10
    selected = np.random.choice(digit_indices, n_samples, replace=False)
    indices_stratificati.extend(selected)

x_te_subset = x_te[np.array(indices_stratificati)]
y_te_subset = mnist_te_labels[np.array(indices_stratificati)]

print(f"Configurazione esperimento robustezza:")
print(f"- Subset stratificato: {len(indices_stratificati)} campioni")
print(f"- Range rumore: {noise_levels[0]:.2f} - {noise_levels[-1]:.2f} (step {noise_levels[1]-noise_levels[0]:.2f})")
print(f"- Livelli testati: {len(noise_levels)}")

# Test robustezza MLP ottimale
print(f"\nTesting robustezza MLP ottimale...")
accuracies_mlp = []

for noise_std in noise_levels:
    x_noisy = add_gaussian_noise(x_te_subset, noise_std)
    acc = MLP_OPTIMAL.score(x_noisy, y_te_subset)
    accuracies_mlp.append(acc)

print("RISULTATI ROBUSTEZZA AL RUMORE:")
print("-" * 40)
print("Noise σ  | MLP Accuratezza")
print("-" * 25)
for noise, acc in zip(noise_levels, accuracies_mlp):
    print(f"{noise:6.2f} |     {acc:.4f}")

# %% [markdown]
# ### Grafico 1: Curve Psicometriche MLP

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Accuratezza assoluta
ax1.plot(noise_levels, accuracies_mlp, 'o-', linewidth=3, markersize=8, 
         color='blue', label='MLP Ottimale', alpha=0.8)

ax1.set_xlabel('Deviazione Standard del Rumore (σ)', fontsize=12)
ax1.set_ylabel('Accuratezza', fontsize=12)
ax1.set_title('Curva Psicometrica: Robustezza al Rumore\nMLP Ottimale', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# Soglia 90%
for i, (noise, acc) in enumerate(zip(noise_levels, accuracies_mlp)):
    if acc < 0.9 and i > 0 and accuracies_mlp[i-1] >= 0.9:
        ax1.axvline(x=noise, color='red', linestyle='--', alpha=0.7)
        ax1.text(noise, 0.92, f'90% threshold\nσ={noise:.2f}', 
                ha='center', va='bottom', fontsize=10, color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        break

# Subplot 2: Degradazione relativa
degradazione_mlp = [(accuracies_mlp[0] - acc) / accuracies_mlp[0] * 100 for acc in accuracies_mlp]

ax2.plot(noise_levels, degradazione_mlp, 'o-', linewidth=3, markersize=8, 
         color='red', label='Degradazione MLP', alpha=0.8)

ax2.set_xlabel('Deviazione Standard del Rumore (σ)', fontsize=12)
ax2.set_ylabel('Degradazione Relativa (%)', fontsize=12)
ax2.set_title('Degradazione Prestazioni\n(% rispetto a condizioni pulite)', fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Grafico 2: Robustezza per Singola Classe

# %%
# Calcolo robustezza per classe
robustezza_per_classe = {}

for digit in range(10):
    mask = y_te_subset == digit
    x_digit = x_te_subset[mask]
    y_digit = y_te_subset[mask]
    
    if len(x_digit) == 0:
        continue
        
    accuracies_digit = []
    for noise_std in noise_levels:
        x_noisy = add_gaussian_noise(x_digit, noise_std)
        y_pred_classes = MLP_OPTIMAL.predict(x_noisy)
        acc = np.mean(y_pred_classes == y_digit)
        accuracies_digit.append(acc)
    
    robustezza_per_classe[digit] = accuracies_digit

fig, ax = plt.subplots(figsize=(12, 8))

colors = plt.cm.tab10(np.linspace(0, 1, 10))
for digit in range(10):
    if digit in robustezza_per_classe:
        ax.plot(noise_levels, robustezza_per_classe[digit], 
                'o-', color=colors[digit], label=f'Cifra {digit}', 
                linewidth=2, markersize=5, alpha=0.8)

ax.set_xlabel('Deviazione Standard del Rumore (σ)', fontsize=12)
ax.set_ylabel('Accuratezza per Classe', fontsize=12)
ax.set_title('Robustezza al Rumore per Singola Classe - MLP Ottimale', fontsize=14)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Grafico 3: Esempio Visivo dell'Effetto del Rumore

# %%
# Esempio visivo progressivo del rumore
esempio_idx = np.where(y_te_subset == 8)[0][0]
esempio_img = x_te_subset[esempio_idx]
noise_demo_levels = [0.0, 0.1, 0.2, 0.3, 0.4]

fig, axes = plt.subplots(1, len(noise_demo_levels), figsize=(15, 3))
fig.suptitle('Effetto Progressivo del Rumore Gaussiano (Cifra 8)', fontsize=14, y=1.05)

for i, noise_std in enumerate(noise_demo_levels):
    if noise_std == 0:
        noisy_img = esempio_img
    else:
        noisy_img = add_gaussian_noise(esempio_img.reshape(1, -1), noise_std)[0]
    
    pred = MLP_OPTIMAL.predict(noisy_img.reshape(1, -1))[0]
    prob = np.max(MLP_OPTIMAL.predict_proba(noisy_img.reshape(1, -1)))
    
    ax = axes[i]
    ax.imshow(noisy_img.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'σ={noise_std:.1f}\nPred:{pred}({prob:.2f})', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Analisi quantitative aggiuntive

# %%
# Analisi soglie critiche (precedenti grafici dettagliati rimossi)
print("ANALISI SOGLIE CRITICHE:")
print("-" * 30)

soglie_accuratezza = [0.95, 0.9, 0.8, 0.7]
for soglia in soglie_accuratezza:
    idx_soglia = np.where(np.array(accuracies_mlp) < soglia)[0]
    if len(idx_soglia) > 0:
        noise_critico = noise_levels[idx_soglia[0]]
        print(f"Soglia {soglia*100:4.0f}%: σ_critico = {noise_critico:.3f}")
    else:
        print(f"Soglia {soglia*100:4.0f}%: Non raggiunta nel range testato")

# Tasso di degradazione
tasso_degradazione = (accuracies_mlp[0] - accuracies_mlp[-1]) / (noise_levels[-1] - noise_levels[0])
print(f"\nTasso degradazione globale: {tasso_degradazione:.4f} acc/σ")

# AUC (Area Under Curve)
auc_robustezza = np.trapz(accuracies_mlp, noise_levels)
print(f"AUC robustezza: {auc_robustezza:.3f}")

# Analisi degradazione per classe
print(f"\nDEGRADAZIONE PER CLASSE (Clean → Final):")
print("-" * 45)
print("Cifra | Clean | Final | Degradazione")
print("-" * 35)

for digit in range(10):
    if digit in robustezza_per_classe:
        clean_acc = robustezza_per_classe[digit][0]
        final_acc = robustezza_per_classe[digit][-1]
        degradazione = clean_acc - final_acc
        print(f"  {digit}   | {clean_acc:.3f} | {final_acc:.3f} |   {degradazione:+.3f}")

# Cifre più/meno robuste
degradazioni_classe = {}
for digit in range(10):
    if digit in robustezza_per_classe:
        degradazioni_classe[digit] = robustezza_per_classe[digit][0] - robustezza_per_classe[digit][-1]

cifra_piu_robusta = min(degradazioni_classe, key=degradazioni_classe.get)
cifra_meno_robusta = max(degradazioni_classe, key=degradazioni_classe.get)

print(f"\nCifra più robusta: {cifra_piu_robusta} (degradazione: {degradazioni_classe[cifra_piu_robusta]:+.3f})")
print(f"Cifra meno robusta: {cifra_meno_robusta} (degradazione: {degradazioni_classe[cifra_meno_robusta]:+.3f})")

# %% [markdown]
# ### Discussione finale e conclusioni Punto C
# **Pattern di degradazione controllata e progressiva:**
# Le curve psicometriche rivelano una degradazione sistematica ma controllata delle prestazioni con soglie critiche ben definite: il modello mantiene **>90% accuratezza fino a σ=0.20**, mentre il collasso significativo inizia oltre **σ=0.35**. Il tasso di degradazione globale (**1.03 acc/σ**) indica resilienza moderata del modello MLP alle perturbazioni gaussiane, con degradazione **non catastrofica** che preserva utilità pratica fino a livelli di rumore sostanziali.
# **Vulnerabilità classe-specifiche critiche:**
# L'analisi per singola classe rivela pattern distintivi di robustezza: la cifra **5** emerge come più robusta (degradazione +0.160) grazie alla sua semplicità strutturale e forme distintive, mentre la cifra **1** risulta più vulnerabile (degradazione +0.970) probabilmente per la dipendenza critica da stroke sottili che il rumore compromette facilmente. Le cifre **0** e **8** mostrano robustezza intermedia nonostante forme chiuse potenzialmente sensibili.
# **Soglie operative per deployment critico:**
# L'identificazione delle soglie operative fornisce riferimenti concreti per deployment: **σ≤0.15** per applicazioni critiche (**>95% accuratezza**), **σ≤0.20** per uso generale (**>90% accuratezza**), **σ≤0.35** per applicazioni tolleranti (**>80% accuratezza**). Oltre **σ=0.40** il modello diventa inaffidabile (**<60% accuratezza**), definendo limiti chiari per condizioni operative accettabili.
# **Allineamento con percezione umana:**
# L'esempio visivo progressivo conferma che il degradation del modello si allinea ragionevolmente con la percezione umana: le predizioni errate emergono quando le immagini diventano genuinamente ambigue anche per osservatori umani (**σ≥0.3**), validando la ragionevolezza del comportamento del modello e suggerendo che i failure modes non sono patologici ma riflettono limitazioni genuine del signal-to-noise ratio.
# **Area Under Curve e resilienza globale:**
# L'**AUC di robustezza (0.366)** quantifica la resilienza globale del modello, fornendo una metrica comparativa per futuri miglioramenti. La degradazione graduale piuttosto che catastrofica suggerisce che il modello ha appreso features relativamente stabili e generali, anche se ulteriori miglioramenti richiederebbero architetture più sofisticate o training con data augmentation specifica per la robustezza al rumore.

# %% [markdown]
# ## Punto D: Effetto della riduzione dei dati di training
# Analizziamo come le prestazioni del modello MLP ottimale degradano quando riduciamo drasticamente la quantità di dati di training disponibili, mantenendo il bilanciamento tra le classi attraverso campionamento stratificato.

# %%
# Configurazione esperimento riduzione dati
train_percentages = [1, 5, 10, 25, 50, 75, 100]
results_data_reduction = []

print("ESPERIMENTO RIDUZIONE DATI DI TRAINING")
print("=" * 50)
print("Architettura: MLP Ottimale (250 neuroni, 1 strato, lr=0.001)")

for percentage in train_percentages:
    print(f"\nTraining con {percentage}% dei dati...")
    
    # Campionamento stratificato per classe
    indices = []
    for digit in range(10):
        digit_indices = np.where(mnist_tr_labels == digit)[0]
        n_digit_samples = int(len(digit_indices) * percentage / 100)
        if n_digit_samples > 0:
            selected_indices = np.random.choice(digit_indices, n_digit_samples, replace=False)
            indices.extend(selected_indices)
    
    indices = np.array(indices)
    x_tr_reduced = x_tr[indices]
    y_tr_reduced = mnist_tr_labels[indices]
    
    # Training MLP ottimale con dati ridotti
    mlp_reduced = crea_mlp_ottimale()
    
    start_time = time.time()
    mlp_reduced.fit(x_tr_reduced, y_tr_reduced)
    training_time = time.time() - start_time
    
    train_acc = mlp_reduced.score(x_tr_reduced, y_tr_reduced)
    test_acc = mlp_reduced.score(x_te, mnist_te_labels)
    
    results_data_reduction.append({
        'percentage': percentage,
        'n_samples': len(indices),
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'overfitting': train_acc - test_acc,
        'training_time': training_time,
        'efficiency': test_acc / training_time
    })
    
    print(f"Samples: {len(indices):5d} | Train: {train_acc:.3f} | Test: {test_acc:.3f} | Time: {training_time:4.1f}s")

# %% [markdown]
# ### Grafico 1: Accuratezza vs Percentuale Dati

# %%
df_reduction = pd.DataFrame(results_data_reduction)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Accuratezza vs percentuale dati
ax1.plot(df_reduction['percentage'], df_reduction['test_accuracy'], 'o-', 
        linewidth=3, markersize=10, color='darkblue', label='Test')
ax1.plot(df_reduction['percentage'], df_reduction['train_accuracy'], 's-', 
        linewidth=3, markersize=10, color='lightblue', label='Train')

ax1.set_xlabel('Percentuale di dati di training utilizzati (%)')
ax1.set_ylabel('Accuratezza')
ax1.set_title('Effetto della riduzione dei dati di training')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Evidenziazione punto 10%
idx_10 = df_reduction[df_reduction['percentage'] == 10].index[0]
ax1.scatter(10, df_reduction.loc[idx_10, 'test_accuracy'], 
          s=200, color='red', zorder=5)
ax1.annotate(f"10%: {df_reduction.loc[idx_10, 'test_accuracy']:.3f}", 
           xy=(10, df_reduction.loc[idx_10, 'test_accuracy']),
           xytext=(20, df_reduction.loc[idx_10, 'test_accuracy'] - 0.05),
           arrowprops=dict(arrowstyle='->', color='red'),
           fontsize=11)

# Subplot 2: Overfitting vs dimensione dataset
ax2.plot(df_reduction['percentage'], df_reduction['overfitting'], 'o-', 
        linewidth=3, markersize=10, color='purple')
ax2.set_xlabel('Percentuale di dati (%)')
ax2.set_ylabel('Overfitting (Train - Test)')
ax2.set_title('Overfitting vs Dimensione dataset')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Grafico 2: Efficienza vs Dimensione Dataset

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Tempo vs dimensione
ax1.plot(df_reduction['n_samples'], df_reduction['training_time'], 'o-', 
        linewidth=3, markersize=10, color='green')
ax1.set_xlabel('Numero di campioni')
ax1.set_ylabel('Tempo di training (s)')
ax1.set_title('Scaling temporale vs Dimensione dataset')
ax1.grid(True, alpha=0.3)

# Subplot 2: Efficienza
ax2.plot(df_reduction['percentage'], df_reduction['efficiency'], 'o-', 
        linewidth=3, markersize=10, color='orange')
ax2.set_xlabel('Percentuale di dati (%)')
ax2.set_ylabel('Efficienza (Accuratezza / Tempo)')
ax2.set_title('Efficienza vs Dimensione dataset')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Analisi quantitative aggiuntive

# %%
# Stampe analisi dettagliate
print("ANALISI SCALING TEMPORALE:")
print("-" * 30)
print("Samples | Time(s) | Scaling")
print("-" * 25)

for i, row in df_reduction.iterrows():
    if i == 0:
        scaling = 1.0
    else:
        scaling = row['training_time'] / df_reduction.iloc[0]['training_time']
    print(f"{int(row['n_samples']):7d} | {row['training_time']:6.1f} | {scaling:6.1f}x")

print(f"\nANALISI OVERFITTING vs DIMENSIONE:")
print("-" * 35)
print("Percentage | Overfitting | Trend")
print("-" * 30)

for i, row in df_reduction.iterrows():
    if row['overfitting'] < 0.02:
        trend = "Basso"
    elif row['overfitting'] < 0.05:
        trend = "Moderato"
    else:
        trend = "Alto"
    print(f"{row['percentage']:9.0f}% | {row['overfitting']:10.3f} | {trend}")

# Punti chiave prestazionali
print(f"\nPUNTI CHIAVE PRESTAZIONALI:")
print("-" * 30)
punto_10 = df_reduction[df_reduction['percentage'] == 10].iloc[0]
punto_100 = df_reduction[df_reduction['percentage'] == 100].iloc[0]

print(f"Con 10% dati: {punto_10['test_accuracy']:.3f} accuratezza ({int(punto_10['n_samples'])} samples)")
print(f"Con 100% dati: {punto_100['test_accuracy']:.3f} accuratezza ({int(punto_100['n_samples'])} samples)")
print(f"Loss prestazionale: {(punto_100['test_accuracy'] - punto_10['test_accuracy'])*100:.1f} punti percentuali")
print(f"Speedup training: {punto_100['training_time']/punto_10['training_time']:.1f}x più veloce con 10%")

# %% [markdown]
# ### Discussione finale e conclusioni Punto D
# **Degradazione prestazionale contenuta e graduale:**
# I risultati empirici confermano robustezza eccezionale alla scarsità di dati: con solo **596 campioni (1%)** il modello raggiunge **84.9% accuratezza**, con **5.996 campioni (10%)** sale a **94.3%**, perdendo solo **3.8 punti percentuali** rispetto alla configurazione completa (98.1% con 60.000 campioni). La progressione è sistematica: **25% → 96.5%**, **50% → 97.6%**, **75% → 97.9%**, dimostrando che l'architettura MLP ottimale mantiene efficacia anche in condizioni di significativa limitazione dati.
# **Scaling temporale quasi-perfetto con benefici pratici immediati:**
# Il tempo di training scala linearmente da **0.2s (1%)** a **31.3s (100%)**, offrendo speedup drammatici per iterazioni rapide: **14.2x più veloce con 10%** mantenendo >94% prestazioni. L'efficienza (accuratezza/tempo) è inversamente proporzionale alla dimensione: **4.25 acc/s con 1%** vs **0.031 acc/s con 100%**, rendendo dataset ridotti ideali per prototipazione dove **2.2 secondi** bastano per 94.3% accuratezza.
# **Paradosso del controllo overfitting con dati limitati:**
# Controintuitivamente, l'overfitting decresce con l'aumento dei dati: **+0.070 (1%)** → **+0.017 (100%)**, indicando calibrazione ottimale tra capacità del modello e complessità del task. Questo comportamento conferma che l'early stopping e la regolarizzazione intrinseca dell'architettura prevengono memorizzazione eccessiva anche con **596 campioni**, suggerendo che la configurazione ottimale identified è genuinamente robusta.
# **Soglie operative evidence-based per deployment reali:**
# I dati definiscono soglie concrete: **25% (14.995 campioni) → 96.5%** per deployment critico con solo **1.6 punti di loss**, **10% (5.996 campioni) → 94.3%** per applicazioni standard con **7.0s training**, **5% (2.996 campioni) → 91.3%** per prototipazione ultra-rapida con **0.8s training**. Queste metriche forniscono guidance quantitativa per bilanciare requirements vs risorse.
# **Implicazioni strategiche per data economics:**
# L'evidenza empirica dimostra ritorni decrescenti massicci: da **25% a 100%** si guadagnano solo **1.6 punti** per **4x più dati** e **4.5x più tempo**. Questo suggerisce che per la maggior parte delle applicazioni pratiche, **dataset del 10-25%** sono sufficienti, riducendo drasticamente **costi di raccolta, labeling e storage** senza compromettere significativamente l'efficacia operativa del sistema.

# %% [markdown]
# ## Punto E: Training con rumore per migliorare la robustezza
# Verifichiamo se l'aggiunta di rumore Gaussiano durante il training può migliorare le prestazioni su dati di test rumorosi, utilizzando l'architettura MLP ottimale e un range esteso di livelli di rumore per data augmentation.

# %%
# Configurazione esperimento training con rumore
training_noise_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
models_with_noise = {}

print("ESPERIMENTO TRAINING CON RUMORE")
print("=" * 40)
print("Architettura: MLP Ottimale (250 neuroni, 1 strato, lr=0.001)")
print(f"Range noise training: 0.0 - 0.3 (step 0.05)")

for train_noise in training_noise_levels:
    print(f"\nTraining con rumore σ = {train_noise}")
    
    # Aggiunta rumore ai dati di training
    if train_noise > 0:
        x_tr_noisy = add_gaussian_noise(x_tr, train_noise)
    else:
        x_tr_noisy = x_tr
    
    # Training MLP ottimale
    mlp_noise = crea_mlp_ottimale()
    
    start_time = time.time()
    mlp_noise.fit(x_tr_noisy, mnist_tr_labels)
    training_time = time.time() - start_time
    
    models_with_noise[train_noise] = mlp_noise
    
    # Test su dati puliti
    clean_acc = mlp_noise.score(x_te, mnist_te_labels)
    print(f"Accuratezza test pulito: {clean_acc:.4f} | Tempo: {training_time:.1f}s")

# Test dei modelli su diversi livelli di rumore nel test set
test_noise_levels = np.arange(0, 0.4, 0.05)
results_noise_training = {}

print(f"\nTest robustezza su range noise 0.0-0.35...")
for train_noise, model in models_with_noise.items():
    accuracies = []
    for test_noise in test_noise_levels:
        x_te_noisy = add_gaussian_noise(x_te_subset, test_noise)
        acc = model.score(x_te_noisy, y_te_subset)
        accuracies.append(acc)
    
    results_noise_training[train_noise] = accuracies
    auc = np.trapz(accuracies, test_noise_levels)
    print(f"Training noise σ={train_noise}: AUC = {auc:.3f}")

# %% [markdown]
# ### Grafico 1: Curve Psicometriche per Diversi Training Noise

# %%
fig, ax = plt.subplots(figsize=(12, 8))

colors = plt.cm.viridis(np.linspace(0, 1, len(training_noise_levels)))

for i, (train_noise, accuracies) in enumerate(results_noise_training.items()):
    ax.plot(test_noise_levels, accuracies, 'o-', 
           label=f'Training σ = {train_noise}',
           color=colors[i], linewidth=2, markersize=6)

ax.set_xlabel('Deviazione standard del rumore (test)', fontsize=12)
ax.set_ylabel('Accuratezza', fontsize=12)
ax.set_title('Effetto del rumore nel training sulla robustezza', fontsize=14)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Grafico 2: AUC vs Training Noise Level

# %%
# Calcolo AUC per ogni livello di training noise
auc_scores = {}
for train_noise, accuracies in results_noise_training.items():
    auc = np.trapz(accuracies, test_noise_levels)
    auc_scores[train_noise] = auc

train_noises = list(auc_scores.keys())
aucs = list(auc_scores.values())

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(train_noises, aucs, 'o-', linewidth=3, markersize=10, color='darkred')
ax.set_xlabel('Rumore nel training (σ)', fontsize=12)
ax.set_ylabel('AUC (Area Under Curve)', fontsize=12)
ax.set_title('Area sotto la curva vs Rumore nel training', fontsize=14)
ax.grid(True, alpha=0.3)

# Identificazione livello ottimale
best_noise = max(auc_scores, key=auc_scores.get)
best_auc = auc_scores[best_noise]
ax.scatter(best_noise, best_auc, s=200, color='gold', zorder=5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Analisi quantitative aggiuntive

# %%
# Analisi miglioramento quantitativo
print("ANALISI MIGLIORAMENTO ROBUSTEZZA:")
print("-" * 40)
print("Train σ | AUC    | vs Clean | Peak Improvement")
print("-" * 45)

baseline_auc = auc_scores[0]
for train_noise in sorted(auc_scores.keys()):
    auc = auc_scores[train_noise]
    improvement = ((auc - baseline_auc) / baseline_auc) * 100
    
    # Trova miglioramento massimo per specifico test noise
    baseline_accs = results_noise_training[0]
    current_accs = results_noise_training[train_noise]
    max_improvement = max([(current_accs[i] - baseline_accs[i]) for i in range(len(current_accs))])
    
    print(f"  {train_noise:4.2f}  | {auc:6.3f} | {improvement:+6.1f}% | {max_improvement:+.3f}")

print(f"\nSOGLIE OTTIMALI TRAINING NOISE:")
print("-" * 35)
print(f"Miglior configurazione: σ = {best_noise}")
print(f"Miglioramento AUC vs baseline: {((best_auc - baseline_auc)/baseline_auc)*100:+.1f}%")

# Analisi soglia efficace
threshold_improvement = 0.02  # 2% miglioramento minimo
effective_noises = [noise for noise, auc in auc_scores.items() 
                   if auc > baseline_auc + threshold_improvement]
if effective_noises:
    print(f"Range efficace (>2% miglioramento): σ = {min(effective_noises):.2f} - {max(effective_noises):.2f}")
else:
    print("Nessun livello supera soglia 2% miglioramento")

# Test specifici per livelli di rumore critici
print(f"\nPRESTAZIONI SU LIVELLI CRITICI:")
print("-" * 35)
critical_test_noises = [0.1, 0.2, 0.3]
for test_noise in critical_test_noises:
    test_idx = int(test_noise / 0.05)
    if test_idx < len(test_noise_levels):
        baseline_acc = results_noise_training[0][test_idx]
        best_acc = results_noise_training[best_noise][test_idx]
        improvement = best_acc - baseline_acc
        print(f"Test σ={test_noise}: {baseline_acc:.3f} → {best_acc:.3f} ({improvement:+.3f})")

# %% [markdown]
# ### Discussione finale e conclusioni Punto E
# **Efficacia comprovata del training con rumore:**
# L'introduzione di rumore Gaussiano durante il training dimostra benefici significativi per la robustezza: la configurazione ottimale (**σ=0.25**) migliora l'AUC di robustezza del **9.3%** rispetto al baseline senza rumore (da 0.308 a 0.337), confermando l'efficacia della data augmentation per la generalizzazione in condizioni avverse. Il peak improvement raggiunge **+0.289** in specifiche condizioni di test noise moderate.
# **Range ottimale di training noise ben definito:**
# Emerge un range efficace **σ=0.15-0.30** dove il rumore fornisce regolarizzazione benefica senza degradare eccessivamente le prestazioni su dati puliti (accuratezza baseline da 98.1% a 96.6% al massimo). Il plateau di prestazioni tra σ=0.20-0.30 indica un meccanismo robusto di regolarizzazione che non richiede fine-tuning estremo del parametro.
# **Miglioramenti concentrati su livelli critici di test noise:**
# I benefici sono particolarmente evidenti su livelli moderati di rumore test: per **σ_test=0.2** l'accuratezza migliora da 0.898 a **0.968 (+0.070)**, mentre per **σ_test=0.3** da 0.812 a **0.962 (+0.149)**. Questo pattern indica che il training con rumore è più efficace nel range di applicazione pratica (**σ=0.1-0.3**) piuttosto che in condizioni estreme o pulite.
# **Meccanismo di regolarizzazione implicita:**
# Il rumore nel training agisce come regolarizzatore implicito, forzando il modello a apprendere **features più robuste** e meno sensibili a perturbazioni locali. La curva AUC vs training noise mostra un optimum chiaro a σ=0.25, suggerendo che esiste un livello ideale di "disturbo controllato" che massimizza la capacità di generalizzazione senza compromettere le prestazioni baseline.
# **Strategia deployment robusto senza modifiche architettoniche:**
# I risultati forniscono una strategia concreta per deployment in ambienti rumorosi: utilizzare training con **σ=0.25** può migliorare significativamente la robustezza con **costo computazionale nullo** (stesso training time). Questa tecnica è particolarmente preziosa per applicazioni dove la qualità dei dati in input può variare (digitalizzazione documenti, acquisizione mobile, trasmissione compressa), offrendo un miglioramento "gratuito" della robustezza senza modifiche architettoniche complesse.

# %% [markdown]
# ## Punto Bonus: Estensione con FashionMNIST e confronto architetturale
# Applichiamo sia l'architettura MLP ottimale che la CNN ottimale al dataset FashionMNIST per valutare la generalizzazione su un task di classificazione più complesso. L'obiettivo è dimostrare che mentre per MNIST un MLP ben calibrato è sufficiente, per task di image recognition più complessi le CNN dovrebbero prevalere grazie ai loro strati convoluzionali.

# %%
# Caricamento e preprocessing FashionMNIST
print("CARICAMENTO FASHIONMNIST")
print("=" * 30)
fashion_tr = FashionMNIST(root="./data", train=True, download=True)
fashion_te = FashionMNIST(root="./data", train=False, download=True)

fashion_tr_data, fashion_tr_labels = fashion_tr.data.numpy(), fashion_tr.targets.numpy()
fashion_te_data, fashion_te_labels = fashion_te.data.numpy(), fashion_te.targets.numpy()

x_fashion_tr = fashion_tr_data.reshape(60000, 28 * 28) / 255.0
x_fashion_te = fashion_te_data.reshape(10000, 28 * 28) / 255.0

# Preprocessing per CNN
x_fashion_tr_conv = fashion_tr_data.reshape(-1, 28, 28, 1) / 255.0
x_fashion_te_conv = fashion_te_data.reshape(-1, 28, 28, 1) / 255.0

fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"FashionMNIST caricato: {x_fashion_tr.shape[0]} train, {x_fashion_te.shape[0]} test")

# Training MLP ottimale su FashionMNIST
print(f"\nTraining MLP ottimale su FashionMNIST...")
mlp_fashion = crea_mlp_ottimale()

start_time = time.time()
mlp_fashion.fit(x_fashion_tr, fashion_tr_labels)
fashion_training_time_mlp = time.time() - start_time

fashion_train_acc_mlp = mlp_fashion.score(x_fashion_tr, fashion_tr_labels)
fashion_test_acc_mlp = mlp_fashion.score(x_fashion_te, fashion_te_labels)

print(f"Training MLP completato in {fashion_training_time_mlp:.1f}s")
print(f"MLP Train accuracy: {fashion_train_acc_mlp:.4f}")
print(f"MLP Test accuracy: {fashion_test_acc_mlp:.4f}")

# Training CNN ottimale su FashionMNIST
print(f"\nTraining CNN ottimale su FashionMNIST...")
cnn_fashion = crea_cnn_ottimale()
early_stopping = keras.callbacks.EarlyStopping(
    patience=5, min_delta=0.001, restore_best_weights=True, verbose=0
)

start_time = time.time()
history_fashion = cnn_fashion.fit(x_fashion_tr_conv, fashion_tr_labels, 
                                 validation_split=0.1, epochs=20, batch_size=128, 
                                 callbacks=[early_stopping], verbose=0)
fashion_training_time_cnn = time.time() - start_time

fashion_train_loss_cnn, fashion_train_acc_cnn = cnn_fashion.evaluate(x_fashion_tr_conv, fashion_tr_labels, verbose=0)
fashion_test_loss_cnn, fashion_test_acc_cnn = cnn_fashion.evaluate(x_fashion_te_conv, fashion_te_labels, verbose=0)

print(f"Training CNN completato in {fashion_training_time_cnn:.1f}s")
print(f"CNN Train accuracy: {fashion_train_acc_cnn:.4f}")
print(f"CNN Test accuracy: {fashion_test_acc_cnn:.4f}")

# Confronto con MNIST
mnist_test_acc_mlp = test_accuracy  # Dalla sezione punto B
mnist_test_acc_cnn = migliore_cnn['test_accuracy']  # Dal punto A

print(f"\nCONFRONTO PRESTAZIONI CROSS-DATASET:")
print("=" * 50)
print(f"MNIST:")
print(f"  MLP Ottimale: {mnist_test_acc_mlp:.4f}")
print(f"  CNN Ottimale: {mnist_test_acc_cnn:.4f}")
print(f"  Gap CNN-MLP: {mnist_test_acc_cnn - mnist_test_acc_mlp:+.4f}")

print(f"\nFashionMNIST:")
print(f"  MLP Ottimale: {fashion_test_acc_mlp:.4f}")
print(f"  CNN Ottimale: {fashion_test_acc_cnn:.4f}")
print(f"  Gap CNN-MLP: {fashion_test_acc_cnn - fashion_test_acc_mlp:+.4f}")

# %% [markdown]
# ### Grafico 1: Confronto Architetturale Cross-Dataset

# %%
# Preparazione dati per il confronto
datasets = ['MNIST', 'FashionMNIST']
mlp_accuracies = [mnist_test_acc_mlp, fashion_test_acc_mlp]
cnn_accuracies = [mnist_test_acc_cnn, fashion_test_acc_cnn]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Confronto accuratezze assolute
x_pos = np.arange(len(datasets))
width = 0.35

bars_mlp = ax1.bar(x_pos - width/2, mlp_accuracies, width, 
                   label='MLP Ottimale', alpha=0.8, color='blue')
bars_cnn = ax1.bar(x_pos + width/2, cnn_accuracies, width, 
                   label='CNN Ottimale', alpha=0.8, color='red')

ax1.set_xlabel('Dataset', fontsize=12)
ax1.set_ylabel('Accuratezza Test', fontsize=12)
ax1.set_title('Confronto Architetturale Cross-Dataset', fontsize=14)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(datasets)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.85, 1.0)

# Annotazioni valori
for i, (mlp_acc, cnn_acc) in enumerate(zip(mlp_accuracies, cnn_accuracies)):
    ax1.annotate(f'{mlp_acc:.3f}', xy=(i - width/2, mlp_acc), 
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    ax1.annotate(f'{cnn_acc:.3f}', xy=(i + width/2, cnn_acc), 
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

# Subplot 2: Gap CNN-MLP
gaps = [cnn_acc - mlp_acc for mlp_acc, cnn_acc in zip(mlp_accuracies, cnn_accuracies)]
colors = ['lightblue' if gap > 0 else 'lightcoral' for gap in gaps]

bars_gap = ax2.bar(datasets, gaps, color=colors, alpha=0.7, width=0.6)
ax2.set_ylabel('Gap CNN - MLP', fontsize=12)
ax2.set_title('Vantaggio CNN vs MLP per Dataset', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

# Annotazioni gap
for i, (dataset, gap) in enumerate(zip(datasets, gaps)):
    ax2.annotate(f'{gap:+.3f}', xy=(i, gap), 
                xytext=(0, 5 if gap > 0 else -15), textcoords="offset points", 
                ha='center', va='bottom' if gap > 0 else 'top', fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Grafico 2: Matrice di Confusione FashionMNIST

# %%
# Calcolo predizioni per FashionMNIST con entrambi i modelli
y_pred_fashion_mlp = mlp_fashion.predict(x_fashion_te)
y_pred_fashion_cnn = cnn_fashion.predict(x_fashion_te_conv)
y_pred_fashion_cnn_classes = np.argmax(y_pred_fashion_cnn, axis=1)

cm_fashion_mlp = metrics.confusion_matrix(fashion_te_labels, y_pred_fashion_mlp)
cm_fashion_cnn = metrics.confusion_matrix(fashion_te_labels, y_pred_fashion_cnn_classes)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Matrice confusione MLP
im1 = ax1.imshow(cm_fashion_mlp, cmap='Blues')
ax1.set_xticks(range(10))
ax1.set_yticks(range(10))
ax1.set_xticklabels([f'{i}' for i in range(10)])
ax1.set_yticklabels([f'{i}: {fashion_classes[i][:8]}' for i in range(10)], fontsize=9)
ax1.set_xlabel('Predetto', fontsize=12)
ax1.set_ylabel('Vero', fontsize=12)
ax1.set_title(f'Matrice Confusione FashionMNIST - MLP\n(Acc: {fashion_test_acc_mlp:.3f})', fontsize=14)

for i in range(10):
    for j in range(10):
        color = 'white' if cm_fashion_mlp[i, j] > cm_fashion_mlp.max() / 2 else 'black'
        ax1.text(j, i, f'{cm_fashion_mlp[i, j]}', ha='center', va='center', 
                color=color, fontweight='bold', fontsize=8)

# Matrice confusione CNN
im2 = ax2.imshow(cm_fashion_cnn, cmap='Reds')
ax2.set_xticks(range(10))
ax2.set_yticks(range(10))
ax2.set_xticklabels([f'{i}' for i in range(10)])
ax2.set_yticklabels([f'{i}: {fashion_classes[i][:8]}' for i in range(10)], fontsize=9)
ax2.set_xlabel('Predetto', fontsize=12)
ax2.set_ylabel('Vero', fontsize=12)
ax2.set_title(f'Matrice Confusione FashionMNIST - CNN\n(Acc: {fashion_test_acc_cnn:.3f})', fontsize=14)

for i in range(10):
    for j in range(10):
        color = 'white' if cm_fashion_cnn[i, j] > cm_fashion_cnn.max() / 2 else 'black'
        ax2.text(j, i, f'{cm_fashion_cnn[i, j]}', ha='center', va='center', 
                color=color, fontweight='bold', fontsize=8)

fig.colorbar(im1, ax=ax1, shrink=0.6)
fig.colorbar(im2, ax=ax2, shrink=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Analisi quantitative aggiuntive

# %%
# Analisi comparative dettagliate
print("ANALISI COMPARATIVE CROSS-DATASET:")
print("=" * 40)

# Calcolo gap di complessità
mnist_complexity_gap = mnist_test_acc_mlp - fashion_test_acc_mlp
fashion_complexity_gap_cnn = mnist_test_acc_cnn - fashion_test_acc_cnn

print("Gap di complessità (MNIST vs FashionMNIST):")
print(f"MLP: {mnist_complexity_gap:+.4f} ({mnist_complexity_gap/fashion_test_acc_mlp*100:+.1f}%)")
print(f"CNN: {fashion_complexity_gap_cnn:+.4f} ({fashion_complexity_gap_cnn/fashion_test_acc_cnn*100:+.1f}%)")

# Analisi vantaggio architetturale
mnist_arch_gap = mnist_test_acc_cnn - mnist_test_acc_mlp
fashion_arch_gap = fashion_test_acc_cnn - fashion_test_acc_mlp

print(f"\nVantaggio CNN vs MLP:")
print(f"MNIST: {mnist_arch_gap:+.4f} ({mnist_arch_gap/mnist_test_acc_mlp*100:+.1f}%)")
print(f"FashionMNIST: {fashion_arch_gap:+.4f} ({fashion_arch_gap/fashion_test_acc_mlp*100:+.1f}%)")
print(f"Amplificazione vantaggio CNN: {fashion_arch_gap/mnist_arch_gap:.1f}x")

# Analisi errori per architettura
mnist_errors_mlp = 10000 - int(mnist_test_acc_mlp * 10000)
fashion_errors_mlp = np.sum(y_pred_fashion_mlp != fashion_te_labels)
fashion_errors_cnn = np.sum(y_pred_fashion_cnn_classes != fashion_te_labels)

print(f"\nANALISI ERRORI ASSOLUTI:")
print("-" * 25)
print(f"MNIST MLP: {mnist_errors_mlp} errori ({(mnist_errors_mlp/10000)*100:.1f}%)")
print(f"FashionMNIST MLP: {fashion_errors_mlp} errori ({(fashion_errors_mlp/10000)*100:.1f}%)")
print(f"FashionMNIST CNN: {fashion_errors_cnn} errori ({(fashion_errors_cnn/10000)*100:.1f}%)")
print(f"Riduzione errori CNN vs MLP su FashionMNIST: {fashion_errors_mlp - fashion_errors_cnn} errori")

# Analisi top confusioni comparative
print(f"\nTOP CONFUSIONI FASHIONMNIST:")
print("-" * 35)

# Top confusioni MLP
fashion_confusion_pairs_mlp = []
for i in range(10):
    for j in range(10):
        if i != j and cm_fashion_mlp[i, j] > 0:
            fashion_confusion_pairs_mlp.append({
                'true_class': fashion_classes[i],
                'pred_class': fashion_classes[j],
                'count': cm_fashion_mlp[i, j],
                'model': 'MLP'
            })

df_fashion_confusion_mlp = pd.DataFrame(fashion_confusion_pairs_mlp)
top_3_fashion_mlp = df_fashion_confusion_mlp.nlargest(3, 'count')

print("Top 3 confusioni MLP:")
for _, row in top_3_fashion_mlp.iterrows():
    print(f"  {row['true_class'][:8]} → {row['pred_class'][:8]}: {row['count']} errori")

# Top confusioni CNN
fashion_confusion_pairs_cnn = []
for i in range(10):
    for j in range(10):
        if i != j and cm_fashion_cnn[i, j] > 0:
            fashion_confusion_pairs_cnn.append({
                'true_class': fashion_classes[i],
                'pred_class': fashion_classes[j],
                'count': cm_fashion_cnn[i, j],
                'model': 'CNN'
            })

df_fashion_confusion_cnn = pd.DataFrame(fashion_confusion_pairs_cnn)
top_3_fashion_cnn = df_fashion_confusion_cnn.nlargest(3, 'count')

print("\nTop 3 confusioni CNN:")
for _, row in top_3_fashion_cnn.iterrows():
    print(f"  {row['true_class'][:8]} → {row['pred_class'][:8]}: {row['count']} errori")

# Analisi efficienza computazionale
print(f"\nANALISI EFFICIENZA COMPUTAZIONALE:")
print("-" * 35)
print(f"Tempo training FashionMNIST:")
print(f"  MLP: {fashion_training_time_mlp:.1f}s")
print(f"  CNN: {fashion_training_time_cnn:.1f}s")
print(f"  Speedup MLP: {fashion_training_time_cnn/fashion_training_time_mlp:.1f}x")

mlp_efficiency_fashion = fashion_test_acc_mlp / fashion_training_time_mlp
cnn_efficiency_fashion = fashion_test_acc_cnn / fashion_training_time_cnn

print(f"\nEfficienza (acc/tempo) FashionMNIST:")
print(f"  MLP: {mlp_efficiency_fashion:.4f} acc/s")
print(f"  CNN: {cnn_efficiency_fashion:.4f} acc/s")
print(f"  Rapporto MLP/CNN: {mlp_efficiency_fashion/cnn_efficiency_fashion:.1f}x")

# %% [markdown]
# ### Discussione finale e conclusioni Punto Bonus
# **Amplificazione empirica del vantaggio CNN su complessità elevata:**
# I risultati quantitativi confermano l'ipotesi architettonica: su MNIST il gap CNN-MLP è marginale (**+0.0089**, +0.9%), mentre su FashionMNIST la CNN raggiunge **90.91%** vs **89.21%** MLP, con gap di **+0.0170** (+1.9%). L'amplificazione del vantaggio CNN è **1.9x**, dimostrando che le architetture convoluzionali diventano essenziali quando la complessità visiva cresce: translation invariance e feature hierarchies catturano pattern tessili e forme di abbigliamento che gli MLP non riescono a discriminare efficacemente.
# **Resilienza differenziale alla complessità del task:**
# FashionMNIST risulta sistematicamente più challenging ma l'impatto varia per architettura: l'MLP perde **8.89 punti percentuali** (98.10% → 89.21%), mentre la CNN perde solo **8.08 punti** (98.99% → 90.91%). Questo 0.8 punti di differenza nella degradazione suggerisce che le CNN sono intrinsecamente più robuste alla complessità incrementale, validando la loro superiorità per image recognition tasks realistici che richiedono discriminazione fine di pattern spaziali complessi.
# **Riduzione errori operativamente significativa:**
# Su FashionMNIST, la CNN riduce gli errori assoluti da **1079 (MLP) a 909 (CNN)**, una diminuzione netta di **170 errori (-15.8%)**. Questo miglioramento non è solo statisticamente significativo ma operativamente rilevante: trasforma un sistema con 10.79% error rate in uno con 9.09% error rate, rappresentando un salto qualitativo concreto per deployment in applicazioni commercial dove ogni punto percentuale conta per user satisfaction e business metrics.
# **Pattern di confusione architettura-specifici rivelatori:**
# L'analisi delle confusioni mostra che entrambe le architetture lottano con distinzioni semanticamente challenging: **Shirt→T-shirt** emerge come confusione top per entrambi (121 MLP, 131 CNN), ma la CNN gestisce meglio confusioni strutturali complesse come **Pullover vs Coat**. Le CNN bilanciano meglio confusioni simmetriche, suggerendo che i meccanismi di pooling e feature extraction gerarchica catturano invarianze morfologiche che l'MLP non apprende efficacemente.
# **Trade-off efficienza-prestazioni quantificato:**
# L'MLP mantiene supremazia computazionale con **3.9x speedup** (34.1s vs 134.2s), ottenendo rispettabili 89.21% con efficienza 0.0262 acc/s vs 0.0068 acc/s CNN. Questo rapporto **3.9:1** rimane favorevole anche su task complessi, rendendo gli MLP competitive per scenari con constraints di velocità severi dove **"good enough accuracy"** giustifica il trade-off prestazioni-efficienza per time-to-market o resource-constrained environments.
# **Validazione empirica di model selection strategy:**
# I dati supportano una strategia evidence-based: **MLP per task semplici** dove overhead CNN non è giustificato dal marginal gain, **CNN per complexity elevata** dove il vantaggio prestazionale (1.7 punti su FashionMNIST) compensa i costi computazionali 4x superiori. Il punto di break-even si situa quando la complessità visiva richiede spatial invariances che solo architetture convoluzionali catturano, validando il principio di architectural complexity matching task intrinsic difficulty.

# %% [markdown]
# ## Conclusioni Generali del Progetto
# ### Riepilogo dei risultati principali
# **Punto A - Analisi Iperparametri:**
# - Identificate architetture ottimali: **MLP(250n_1S_lr0.001)** con **98.10%** e **CNN(extended_lr0.001)** con **98.85%**
# - Learning rate critico: range ottimale **0.001-0.01**, collasso catastrofico a **0.1** (drop a 86.1%)
# - Profondità controproducente: **1 strato supera 2 strati** di +2.2 punti su MNIST per overfitting control
# - Efficienza dominata da MLP: rapporto **12.4x** favorevole per accuratezza/tempo vs CNN
# **Punto B - Cifre difficili:**
# - Gerarchia difficoltà: **8(2.8%), 2(2.5%), 5(2.4%)** vs **0(<1%), 1(<1%)** con pattern morfologici giustificati
# - Pattern confusione top: **4→9, 7→2, 8→3** riflettono similitudini strutturali genuine
# - Calibrazione confidenza eccellente: correlazione **r=0.774** per quality control automatico
# - **190 errori totali** su 10K (1.90%) con distribuzione controllata e concentrata
# **Punto C - Robustezza al rumore:**
# - Soglie operative: **σ≤0.15(>95%), σ≤0.20(>90%), σ≤0.35(>80%)** per deployment stratificato
# - Degradazione controllata: tasso **1.03 acc/σ** senza collasso catastrofico
# - Vulnerabilità classe-specifiche: **cifra 1 più vulnerabile (+0.970), cifra 5 più robusta (+0.160)**
# - **AUC robustezza 0.366** quantifica resilienza globale del modello
# **Punto D - Riduzione dati:**
# - Robustezza a scarsità: **10% dati → 94.3% accuratezza** (-3.8 punti con speedup 14.2x)
# - Scaling quasi-lineare con efficiency massima per dataset ridotti (**4.25 acc/s** con 1% vs **0.031** con 100%)
# - Overfitting paradossalmente controllato con dati limitati grazie a regolarizzazione intrinseca
# - Soglie operative: **25%(>96.5%), 10%(>94.3%), 5%(>91.3%)** per bilanciamento prestazioni-risorse
# **Punto E - Training con rumore:**
# - Data augmentation efficace: **σ=0.25 ottimale** con **+9.4% AUC** improvement vs baseline clean
# - Range efficace **σ=0.15-0.30** per regolarizzazione senza degradazione baseline significativa
# - Benefici concentrati su test noise moderato: **σ=0.2 (+0.071), σ=0.3 (+0.149)** improvement
# - Strategia deployment robusto **costo-zero** senza modifiche architettoniche
# **Punto Bonus - Confronto Architetturale:**
# - **Amplificazione 1.9x vantaggio CNN** su complexity elevata: MNIST(+0.9%) → FashionMNIST(+1.9%)
# - CNN più robusta a complessità: drop **8.08 punti** vs **8.89 punti MLP** su FashionMNIST
# - **Riduzione 170 errori (-15.8%)** CNN vs MLP su FashionMNIST con miglioramento operativo concreto
# - Validazione model selection: **MLP per task semplici, CNN per complexity elevata**
# ### Insights metodologici trasversali fondamentali
# **Principio di Parsimonia Architettonica:**
# La ricerca sistematica conferma che per task visivi semplici come MNIST, **architetture snelle e ben calibrate superano configurazioni complesse**. Il learning rate emerge come iperparametro più critico, mentre profondità aggiuntiva introduce overfitting senza benefici. Questo validata il principio che complexity should match task intrinsic difficulty.
# **Robustezza Intrinseca e Generalizzazione:**
# I modelli dimostrano **resilienza intrinseca** a condizioni avverse (rumore, dati limitati) quando l'architettura è appropriata al task. La data augmentation con rumore controllato fornisce miglioramenti significativi "gratuiti" senza costi architettonali, evidenziando l'importanza di **regolarizzazione intelligente** vs brute-force complexity.
# **Trade-off Efficiency-Performance Ottimizzabile:**
# Il rapporto accuratezza-tempo rivela che per molte applicazioni pratiche, **configurazioni moderate offrono optimal value**: MLP(100, lr=0.01) raggiunge 97.3% in <10 secondi, ideale per prototipazione. Questo dimostra che **deployment pragmatico** spesso non richiede architetture state-of-the-art.
# **Auto-Calibrazione e Reliability Engineering:**
# I modelli mostrano **eccellente autoconsapevolezza** attraverso calibrazione delle confidenze, fornendo meccanismi naturali di quality control per deployment critico. Soglie confidence-based permettono **escalation automatica** senza overhead computazionale.
# ### Raccomandazioni strategiche per applicazioni reali
# **Per sviluppo rapido e prototipazione:**
# - **MLP(100, lr=0.01)** per iterazione veloce e proof-of-concept con >97% performance
# - **Dataset 10-25%** per validazione iniziale mantenendo >94% prestazioni con speedup massicci
# - **Training noise σ=0.25** per robustezza immediata e regolarizzazione automatica
# **Per deployment critico e production:**
# - **MLP(250, lr=0.001)** per maximum bilanciamento prestazioni-efficienza su task semplici
# - **Soglie confidence <0.80** per escalation manuale, **>0.95** per full automation
# - **Validazione cross-domain obbligatoria** prima del deployment per robustezza garantita
# **Per massimizzazione prestazioni su task complessi:**
# - **CNN extended** quando task complexity giustifica overhead computazionale significativo
# - **Architecture selection data-driven**: complexity matching basato su empirical validation
# - **Data augmentation sistematica** per robustezza operativa in ambienti variabili
# ### Contributo metodologico e direzioni future
# Questo studio dimostra l'importanza di **systematic empirical validation** per AI deployment decisions. La metodologia di confronto sistematico tra architetture, combined with pragmatic efficiency analysis, fornisce un framework replicabile per **evidence-based model selection** in contesti applicativi reali.
# Le direzioni future includono estensione a **ensemble methods** per error reduction oltre i single-model limits, **neural architecture search** per automatic optimization, e **uncertainty quantification** per deployment safety in critical applications.