# %% [markdown]
#  # Mini Progetto Intelligenza Artificiale - Riconoscimento cifre manoscritte
# 
# 
# 
#  **Nome:** Giulio
# 
#  **Cognome:** Bottacin
# 
#  **Matricola:** 2042340
# 
#  **Data consegna:** 5/6/2025
# 
# 
# 
#  ## Obiettivo
# 
# 
# 
#  In questo progetto esploreremo il riconoscimento di cifre manoscritte utilizzando il dataset MNIST, implementando simulazioni per studiare come diversi fattori influenzano le prestazioni dei modelli di deep learning. Analizzeremo in particolare l'impatto degli iperparametri, la robustezza al rumore e l'effetto della quantità di dati di training.

# %% [markdown]
#  ## Importazione delle librerie necessarie

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
#  ## Funzioni Helper Globali

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
#  ## Caricamento e preparazione del dataset MNIST

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
#  ## Punto A: Effetto degli iperparametri sulle prestazioni
# 
# 
# 
#  Analizziamo sistematicamente come variano le prestazioni dei modelli MLP e CNN al variare degli iperparametri chiave. Confronteremo 18 configurazioni MLP e 6 configurazioni CNN per un totale di 24 esperimenti mirati.
# 
#  Confronteremo 18 configurazioni MLP e 6 configurazioni CNN per un totale di 24 esperimenti mirati.

# %% [markdown]
#  ### Configurazione esperimenti sistematici
# 
# 
# 
#  ***MLP (18 esperimenti):***
# 
#  - **Neuroni per strato**: *50, 100, 250* per testare la copertura da reti piccole a medio-grandi
# 
#  - **Numero layers**: *1 vs 2* strati nascosti per fare il confronto profondità vs larghezza
# 
#  - **Learning rate**: *0.001, 0.01, 0.1*
# 
# 
# 
#  ***CNN (6 esperimenti):***
# 
#  - **Filtri**: *32*, standard per MNIST, computazionalmente efficiente
# 
#  - **Architettura**: *baseline vs extended* per fare il confronto sulla complessità
# 
#  - **Learning rate**: *0.001, 0.01, 0.1*
# 
# 
# 
#  Per entrambi i modelli si è scelto di utilizzare il solver **Adam**, ormai standard e più performante di SDG.
# 
#  Si è volutamente scelto di eseguire meno esperimenti sulle CNN in quanto richiedono tempi molto più lunghi di training rispetto alle MLP.
# 
# 
# 
#  #### Scelta dei parametri di training
# 
# 
# 
#  ***MLP:***
# 
#  - *max_iter = 100* è sufficiente per convergenza su MNIST basato su cifre manoscritte.
# 
#  - *early_stopping = True*, previene l'overfitting essenziale quando sono presenti molti parametri.
# 
#  - *validation_fraction = 0.1*, split standard 90/10.
# 
#  - *tol = 0.001* è una precisione ragionevole per classificazione.
# 
#  - *n_iter_no_change = 10* è un livello di pazienza adeguata per permettere oscillazioni temporanee.
# 
# 
# 
#  ***CNN:***
# 
#  - *epochs = 20* valore di compromesso per bilanciare velocità e convergenza, il valore è più basso delle MLP perchè le CNN tipicamente convergono più velocemente.
# 
#  - *batch_size = 128*, trade-off memoria/velocità ottimale per dataset size.
# 
#  - *validation_split = 0.1*, coerente con le scelte di MLP.
# 
#  - *patience = 5*, le CNN sono meno soggette a oscillazioni quindi è stato scelto un livello di pazienza minore.
# 
#  - *min_delta = 0.001*, scelta la stessa precisione degli MLP per comparabilità diretta.
# 
# 
# 
#  Questa configurazione permette un confronto sistematico e bilanciato tra i due tipi di architetture.

# %% [markdown]
#  #### Funzioni helper per stampe risultati

# %%
def stampa_header_esperimento(num_esp, totale, tipo_modello, config):
    print(f"\n[{num_esp:2d}/{totale}] {tipo_modello}: {config}")
    print("-" * 50)

def stampa_risultati_esperimento(risultati):
    print(f"Accuracy Training: {risultati['train_accuracy']:.4f} | Accuracy Test: {risultati['test_accuracy']:.4f}")
    print(f"Tempo: {risultati['training_time']:6.1f}s | Iterazioni: {risultati['iterations']:3d}")
    print(f"Overfitting: {risultati['overfitting']:+.4f}")


# %% [markdown]
#  ### Esperimenti sistematici MLP e CNN

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
#  ### Grafico 1: Effetto del Learning Rate sulle prestazioni MLP

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
#  ### Grafico 2: Confronto Completo delle Architetture

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

ax.annotate(f'Miglior MLP\n{migliore_mlp["test_accuracy"]:.4f}', 
           xy=(idx_migliore_mlp + larghezza/2, migliore_mlp['test_accuracy']),
           xytext=(10, 20), textcoords='offset points',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
           arrowprops=dict(arrowstyle='->', color='blue'))

ax.annotate(f'Miglior CNN\n{migliore_cnn["test_accuracy"]:.4f}', 
           xy=(idx_migliore_cnn + larghezza/2, migliore_cnn['test_accuracy']),
           xytext=(10, -30), textcoords='offset points',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
           arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()
plt.show()


# %% [markdown]
#  ### Grafico 3: Effetto Scaling MLP (1 vs 2 Strati Nascosti)

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
#  ### Analisi quantitative aggiuntive e stampe risultati

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
#  ### Discussione finale e conclusioni Punto A
# 
# 
# 
#  **Architetture ottimali identificate:**
# 
# 
# 
#  Gli esperimenti sistematici su 24 configurazioni hanno identificato due architetture leader:
# 
#  - **MLP** con 250 neuroni, 1 strato nascosto e learning rate 0.001 raggiunge **98.10%** di accuratezza
# 
#  - **CNN** extended con learning rate 0.001 ottiene **98.85%** stabilendo il benchmark prestazionale
# 
# 
# 
#  **Insights critici sul Learning Rate:**
# 
# 
# 
#  Il learning rate emerge come iperparametro decisivo: 0.001-0.01 costituisce il range ottimale con 0.001 che maximizza l'accuratezza (97.65% media) mentre 0.01 offre il miglior compromesso velocità-prestazioni (97.32%). Learning rate 0.1 causa collasso prestazionale drammatico (86.12% per MLP), evidenziando l'importanza critica della calibrazione.
# 
# 
# 
#  **Profondità vs Larghezza negli MLP:**
# 
# 
# 
#  Controintuitivamente, le architetture a 1 strato superano sistematicamente quelle a 2 strati con vantaggio medio di +2.2 punti percentuali, indicando che su MNIST la maggiore profondità introduce overfitting piuttosto che benefici. Questo suggerisce che la complessità intrinseca del task non giustifica architetture profonde.
# 
# 
# 
#  **Efficienza computazionale:**
# 
# 
# 
#  Gli MLP dominano l'efficienza (0.095 vs 0.018 acc/s) con rapporto 5.3x favorevole, principalmente per tempi di training drammaticamente inferiori che compensano il gap di accuratezza. Le configurazioni MLP piccole (50-100 neuroni, LR=0.01) emergono ideali per prototipazione rapida.
# 
# 
# 
#  **Robustezza all'overfitting:**
# 
# 
# 
#  Le CNN mostrano controllo superiore dell'overfitting (range 0.0012-0.0114) rispetto agli MLP (0.0004-0.0201) grazie ai meccanismi intrinseci di regolarizzazione. Sorprendentemente, la correlazione parametri-overfitting è debole (r=0.31), evidenziando che l'architettura conta più della complessità assoluta.
# 
# 
# 
#  **Raccomandazioni strategiche:**
# 
# 
# 
#  - Per **deployment critico**: MLP(250, lr=0.001) bilancia 98.1% accuratezza con efficienza 4x superiore alle CNN
# 
#  - Per **prototipazione veloce**: MLP(100, lr=0.01) offre 97.3% accuratezza in <10 secondi
# 
#  - Per **massimizzazione prestazioni**: CNN extended con lr=0.001 quando il costo computazionale è giustificabile

# %% [markdown]
#  ## Punto B: Analisi delle cifre più difficili da riconoscere
# 
# 
# 
#  Utilizziamo l'architettura MLP ottimale identificata nel Punto A per analizzare sistematicamente quali cifre sono più difficili da classificare attraverso la matrice di confusione e l'analisi degli errori.

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
#  ### Grafico 1: Matrice di Confusione

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
#  ### Grafico 2: Difficoltà di Riconoscimento per Cifra

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
#  ### Analisi quantitative aggiuntive

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
#  ### Discussione finale e conclusioni Punto B
# 
# 
# 
#  **Gerarchia di difficoltà identificata:**
# 
# 
# 
#  L'analisi quantitativa rivela una stratificazione clara delle cifre per difficoltà con la cifra **8** che emerge come più problematica (2.8% errori), seguita da **2** (2.5%) e **5** (2.4%), mentre **0** e **1** si confermano più robuste (<1% errori). Questa distribuzione riflette la complessità morfologica intrinseca: la cifra 8 presenta due loop chiusi che creano ambiguità con 3, 6 o 9.
# 
# 
# 
#  **Pattern di confusione sistematici:**
# 
# 
# 
#  Le Top 3 confusioni (4→9, 7→2, 8→3) rivelano errori morfologicamente giustificati dove la similitudine strutturale induce il modello in errore. L'analisi della simmetria mostra pattern direzionali significativi: 2↔3 presenta simmetria elevata (bidirezionale), mentre 8→3 mostra forte asimmetria indicando vulnerabilità specifica del modello nell'interpretare i loop della cifra 8.
# 
# 
# 
#  **Meccanismo di calibrazione della confidenza:**
# 
# 
# 
#  Il modello dimostra eccellente autoconsapevolezza con confidenze elevate per predizioni corrette (0.990-0.998) e significativamente ridotte per errori (0.722-0.845). La correlazione confidenza-accuratezza (r=0.78) fornisce un meccanismo naturale di early warning: soglie <0.80 potrebbero attivare controlli manuali, mentre >0.95 garantiscono affidabilità quasi assoluta.
# 
# 
# 
#  **Concentrazione e distribuzione degli errori:**
# 
# 
# 
#  Con solo 190 errori su 10.000 esempi (1.90% globale), il modello mostra robustezza generale. Le Top 3 confusioni rappresentano 22 errori (11.6% del totale), indicando vulnerabilità moderate senza pattern critici localizzati. Gli errori residui coinvolgono casi di genuina ambiguità morfologica dove anche osservatori umani potrebbero esitare.
# 
# 
# 
#  **Implicazioni per il miglioramento:**
# 
# 
# 
#  I risultati suggeriscono che ulteriori guadagni richiederanno interventi mirati: data augmentation per cifre problematiche (8, 2, 5), fine-tuning su confusioni specifiche, o architetture convoluzionali per catturare invarianze spaziali più sofisticate, dato che gli errori residui rappresentano il limite naturale per questo livello di complessità architettonica.

# %% [markdown]
#  ## Punto C: Curve psicometriche - Effetto del rumore
# 
# 
# 
#  Analizziamo sistematicamente come l'accuratezza di riconoscimento degrada all'aumentare del rumore Gaussiano aggiunto alle immagini di test, utilizzando l'architettura MLP ottimale per valutare la robustezza intrinseca del modello.

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
#  ### Grafico 1: Curve Psicometriche MLP

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
#  ### Grafico 2: Robustezza per Singola Classe

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
#  ### Grafico 3: Esempio Visivo dell'Effetto del Rumore

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
#  ### Analisi quantitative aggiuntive

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
#  ### Discussione finale e conclusioni Punto C
# 
# 
# 
#  **Pattern di degradazione sistematici:**
# 
# 
# 
#  Le curve psicometriche rivelano una degradazione progressiva ma controllata delle prestazioni con soglie critiche ben definite: il modello mantiene >90% accuratezza fino a σ=0.20, mentre il collasso significativo inizia oltre σ=0.35. Il tasso di degradazione globale (1.01 acc/σ) indica resilienza moderata del modello MLP alle perturbazioni gaussiane.
# 
# 
# 
#  **Vulnerabilità classe-specifiche:**
# 
# 
# 
#  L'analisi per singola classe rivela pattern distintivi di robustezza: la cifra **1** emerge come più robusta (degradazione +0.785) grazie alla sua semplicità strutturale, mentre la cifra **4** risulta più vulnerabile (degradazione +0.89) probabilmente per la complessità delle connessioni angolari che il rumore altera criticamente. Le cifre **0** e **8** mostrano robustezza intermedia nonostante forme chiuse potenzialmente sensibili.
# 
# 
# 
#  **Soglie critiche per deployment:**
# 
# 
# 
#  L'identificazione delle soglie operative fornisce riferimenti cruciali: σ≤0.15 per applicazioni critiche (>95% accuratezza), σ≤0.20 per uso generale (>90% accuratezza), σ≤0.35 per applicazioni tolleranti (>80% accuratezza). Oltre σ=0.40 il modello diventa inaffidabile (<60% accuratezza).
# 
# 
# 
#  **Allineamento con percezione umana:**
# 
# 
# 
#  L'esempio visivo conferma che il degradation del modello si allinea ragionevolmente con la percezione umana: le predizioni errate emergono quando le immagini diventano genuinamente ambigue anche per osservatori umani (σ≥0.3), validando la ragionevolezza del comportamento del modello.
# 
# 
# 
#  **Area Under Curve e resilienza globale:**
# 
# 
# 
#  L'AUC di robustezza (0.368) quantifica la resilienza globale del modello, fornendo una metrica comparativa per futuri miglioramenti. La degradazione graduale piuttosto che catastrofica suggerisce che il modello ha appreso features relativamente stabili, anche se ulteriori miglioramenti richiederebbero architetture più sofisticate o training con data augmentation specifica per la robustezza al rumore.

# %% [markdown]
#  ## Punto D: Effetto della riduzione dei dati di training
# 
# 
# 
#  Analizziamo come le prestazioni del modello MLP ottimale degradano quando riduciamo drasticamente la quantità di dati di training disponibili, mantenendo il bilanciamento tra le classi attraverso campionamento stratificato.

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
#  ### Grafico 1: Accuratezza vs Percentuale Dati

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
#  ### Grafico 2: Efficienza vs Dimensione Dataset

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
#  ### Analisi quantitative aggiuntive

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
    print(f"{row['n_samples']:7d} | {row['training_time']:6.1f} | {scaling:6.1f}x")

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

print(f"Con 10% dati: {punto_10['test_accuracy']:.3f} accuratezza ({punto_10['n_samples']} samples)")
print(f"Con 100% dati: {punto_100['test_accuracy']:.3f} accuratezza ({punto_100['n_samples']} samples)")
print(f"Loss prestazionale: {(punto_100['test_accuracy'] - punto_10['test_accuracy'])*100:.1f} punti percentuali")
print(f"Speedup training: {punto_100['training_time']/punto_10['training_time']:.1f}x più veloce con 10%")


# %% [markdown]
#  ### Discussione finale e conclusioni Punto D
# 
# 
# 
#  **Degradazione prestazionale sistematica:**
# 
# 
# 
#  La riduzione dei dati mostra un impatto significativo ma graduale sulle prestazioni: con solo il 10% dei dati (5.996 campioni) il modello raggiunge comunque 94.45% di accuratezza, perdendo solo 3.4 punti percentuali rispetto alla configurazione completa (97.84%). Questo dimostra la robustezza intrinseca dell'architettura MLP ottimale anche in condizioni di scarsità dati.
# 
# 
# 
#  **Scaling temporale ed efficienza:**
# 
# 
# 
#  Il tempo di training scala quasi linearmente con la dimensione del dataset (da 3.5s con 1% a 28.1s con 100%), offrendo un trade-off attraente per applicazioni con vincoli temporali. L'efficienza (accuratezza/tempo) è massima con dataset ridotti, suggerendo che per prototipazione rapida o proof-of-concept, dataset del 10-25% possono essere ottimali.
# 
# 
# 
#  **Controllo dell'overfitting con dati limitati:**
# 
# 
# 
#  Controintuitivamente, dataset più piccoli mostrano overfitting ridotto (1% dati: +0.073, 100% dati: +0.017), indicando che la capacità del modello è ben calibrata rispetto alla complessità del task. Questo comportamento suggerisce che il modello non soffre di memorizzazione eccessiva anche con dati limitati.
# 
# 
# 
#  **Soglie operative per applicazioni reali:**
# 
# 
# 
#  I risultati identificano soglie pratiche: 25% dati (15K campioni) per >95% accuratezza in applicazioni critiche, 10% dati (6K campioni) per >94% accuratezza in scenari con vincoli di velocità, 5% dati (3K campioni) per >92% accuratezza in prototipazione rapida. Queste soglie forniscono linee guida concrete per bilanciare prestazioni e risorse computazionali.
# 
# 
# 
#  **Implicazioni per data collection e deployment:**
# 
# 
# 
#  La robustezza del modello con dati ridotti ha implicazioni significative per strategie di raccolta dati: investimenti massicci in dataset potrebbero fornire ritorni marginali decrescenti, mentre dataset moderati (10-25% della scala completa) potrebbero essere sufficienti per molte applicazioni pratiche, riducendo significativamente costi e tempi di sviluppo.

# %% [markdown]
#  ## Punto E: Training con rumore per migliorare la robustezza
# 
# 
# 
#  Verifichiamo se l'aggiunta di rumore Gaussiano durante il training può migliorare le prestazioni su dati di test rumorosi, utilizzando l'architettura MLP ottimale e un range esteso di livelli di rumore per data augmentation.

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
#  ### Grafico 1: Curve Psicometriche per Diversi Training Noise

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
#  ### Grafico 2: AUC vs Training Noise Level

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
ax.annotate(f'Ottimo: σ={best_noise}\nAUC={best_auc:.3f}', 
           xy=(best_noise, best_auc),
           xytext=(best_noise + 0.05, best_auc + 0.01),
           arrowprops=dict(arrowstyle='->', color='gold'),
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()


# %% [markdown]
#  ### Analisi quantitative aggiuntive

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
#  ### Discussione finale e conclusioni Punto E
# 
# 
# 
#  **Efficacia del training con rumore:**
# 
# 
# 
#  L'introduzione di rumore Gaussiano durante il training dimostra benefici significativi per la robustezza: la configurazione ottimale (σ=0.15) migliora l'AUC di robustezza del 8.7% rispetto al baseline senza rumore (da 0.331 a 0.360), confermando l'efficacia della data augmentation per la generalizzazione in condizioni avverse.
# 
# 
# 
#  **Range ottimale di training noise:**
# 
# 
# 
#  Emerge un range efficace σ=0.10-0.20 dove il rumore fornisce regolarizzazione benefica senza degradare eccessivamente le prestazioni su dati puliti. Oltre σ=0.25 i benefici si riducono e le prestazioni baseline iniziano a soffrire, indicando un trade-off ottimale ben definito.
# 
# 
# 
#  **Miglioramenti per livelli critici di test noise:**
# 
# 
# 
#  I benefici sono particolarmente evidenti su livelli moderati di rumore test: per σ_test=0.2 l'accuratezza migliora da 0.817 a 0.876 (+0.059), mentre per σ_test=0.3 da 0.739 a 0.821 (+0.082). Questo pattern indica che il training con rumore è più efficace nel range di applicazione pratica piuttosto che in condizioni estreme.
# 
# 
# 
#  **Meccanismo di regolarizzazione:**
# 
# 
# 
#  Il rumore nel training agisce come regolarizzatore implicito, forzando il modello a apprendere features più robuste e meno sensibili a perturbazioni locali. La curva AUC vs training noise mostra un optimum chiaro, suggerendo che esiste un livello ideale di "disturbo controllato" che massimizza la capacità di generalizzazione.
# 
# 
# 
#  **Implicazioni per deployment robusto:**
# 
# 
# 
#  I risultati forniscono una strategia concreta per deployment in ambienti rumorosi: utilizzare training con σ=0.15 può migliorare significativamente la robustezza con costo computazionale minimo. Questa tecnica è particolarmente preziosa per applicazioni dove la qualità dei dati in input può variare (digitalizzazione documenti, acquisizione mobile, trasmissione compressa), offrendo un miglioramento "gratuito" della robustezza senza modifiche architettoniche.

# %% [markdown]
#  ## Punto Bonus: Estensione con FashionMNIST
# 
# 
# 
#  Applichiamo l'architettura MLP ottimale al dataset FashionMNIST per valutare la generalizzazione su un task di classificazione più complesso e confrontare le prestazioni con MNIST per analizzare l'effetto della complessità del dataset.

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

fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"FashionMNIST caricato: {x_fashion_tr.shape[0]} train, {x_fashion_te.shape[0]} test")

# Training MLP ottimale su FashionMNIST
print(f"\nTraining MLP ottimale su FashionMNIST...")
mlp_fashion = crea_mlp_ottimale()

start_time = time.time()
mlp_fashion.fit(x_fashion_tr, fashion_tr_labels)
fashion_training_time = time.time() - start_time

fashion_train_acc = mlp_fashion.score(x_fashion_tr, fashion_tr_labels)
fashion_test_acc = mlp_fashion.score(x_fashion_te, fashion_te_labels)

print(f"Training completato in {fashion_training_time:.1f}s")
print(f"Train accuracy: {fashion_train_acc:.4f}")
print(f"Test accuracy: {fashion_test_acc:.4f}")
print(f"Overfitting: {fashion_train_acc - fashion_test_acc:+.4f}")

# Confronto diretto con MNIST
mnist_test_acc = test_accuracy  # Dalla sezione punto B
print(f"\nCONFRONTO PRESTAZIONI:")
print(f"MNIST test accuracy: {mnist_test_acc:.4f}")
print(f"FashionMNIST test accuracy: {fashion_test_acc:.4f}")
print(f"Gap di complessità: {mnist_test_acc - fashion_test_acc:+.4f} ({((mnist_test_acc - fashion_test_acc)/fashion_test_acc)*100:+.1f}%)")


# %% [markdown]
#  ### Grafico 1: Confronto MNIST vs FashionMNIST

# %%
# Accuratezza per classe su entrambi i dataset
mnist_class_accs = []
fashion_class_accs = []

# MNIST accuratezze per classe (riutilizzo da punto B)
for digit in range(10):
    mask_m = mnist_te_labels == digit
    acc_m = np.sum((y_pred == mnist_te_labels) & mask_m) / np.sum(mask_m)
    mnist_class_accs.append(acc_m)

# FashionMNIST accuratezze per classe
y_pred_fashion = mlp_fashion.predict(x_fashion_te)
for digit in range(10):
    mask_f = fashion_te_labels == digit
    acc_f = np.sum((y_pred_fashion == fashion_te_labels) & mask_f) / np.sum(mask_f)
    fashion_class_accs.append(acc_f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Confronto accuratezze per classe
x_pos = np.arange(10)
width = 0.35

bars_mnist = ax1.bar(x_pos - width/2, mnist_class_accs, width, 
                    label='MNIST', alpha=0.8, color='blue')
bars_fashion = ax1.bar(x_pos + width/2, fashion_class_accs, width, 
                      label='FashionMNIST', alpha=0.8, color='red')

ax1.set_xlabel('Classe', fontsize=12)
ax1.set_ylabel('Accuratezza per classe', fontsize=12)
ax1.set_title('Confronto Accuratezza per Classe:\nMNIST vs FashionMNIST', fontsize=14)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'{i}' for i in range(10)])
ax1.legend()
ax1.grid(True, alpha=0.3)

# Annotazioni differenze significative
for i, (mnist_acc, fashion_acc) in enumerate(zip(mnist_class_accs, fashion_class_accs)):
    if abs(mnist_acc - fashion_acc) > 0.05:  # Differenza > 5%
        ax1.annotate(f'{mnist_acc - fashion_acc:+.2f}', 
                    xy=(i, max(mnist_acc, fashion_acc) + 0.02),
                    ha='center', fontsize=9, color='darkred', fontweight='bold')

# Subplot 2: Accuratezza globale confronto
datasets = ['MNIST', 'FashionMNIST']
accuracies = [mnist_test_acc, fashion_test_acc]
colors = ['blue', 'red']

bars = ax2.bar(datasets, accuracies, color=colors, alpha=0.7, width=0.6)
ax2.set_ylabel('Accuratezza Test Globale', fontsize=12)
ax2.set_title('Confronto Prestazioni Globali', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.8, 1.0)

# Annotazioni valori
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax2.annotate(f'{acc:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                fontsize=12, fontweight='bold')

# Annotazione gap
gap = accuracies[0] - accuracies[1]
ax2.annotate(f'Gap: {gap:.3f}\n({gap/accuracies[1]*100:+.1f}%)', 
            xy=(0.5, (accuracies[0] + accuracies[1])/2),
            ha='center', fontsize=11, color='darkgreen', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.show()


# %% [markdown]
#  ### Grafico 2: Matrice di Confusione FashionMNIST

# %%
cm_fashion = metrics.confusion_matrix(fashion_te_labels, y_pred_fashion)

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(cm_fashion, cmap='Blues')
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels([f'{i}' for i in range(10)])
ax.set_yticklabels([f'{i}: {fashion_classes[i][:8]}' for i in range(10)], fontsize=10)
ax.set_xlabel('Predetto', fontsize=12)
ax.set_ylabel('Vero', fontsize=12)
ax.set_title('Matrice di Confusione - FashionMNIST', fontsize=14)

# Annotazioni con valori
for i in range(10):
    for j in range(10):
        color = 'white' if cm_fashion[i, j] > cm_fashion.max() / 2 else 'black'
        ax.text(j, i, f'{cm_fashion[i, j]}', ha='center', va='center', 
                color=color, fontweight='bold')

fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()


# %% [markdown]
#  ### Analisi quantitative aggiuntive

# %%
# Analisi robustezza comparativa (precedenti curve psicometriche rimosse)
print("ANALISI ROBUSTEZZA COMPARATIVA:")
print("-" * 35)

# Test robustezza su subset FashionMNIST
x_fashion_te_subset = x_fashion_te[:2000]
y_fashion_te_subset = fashion_te_labels[:2000]

noise_levels_comp = np.arange(0, 0.3, 0.05)
acc_mnist_comp = []
acc_fashion_comp = []

for noise_std in noise_levels_comp:
    # MNIST
    x_noisy_mnist = add_gaussian_noise(x_te_subset, noise_std)
    acc_mnist_comp.append(MLP_OPTIMAL.score(x_noisy_mnist, y_te_subset))
    
    # FashionMNIST  
    x_noisy_fashion = add_gaussian_noise(x_fashion_te_subset, noise_std)
    acc_fashion_comp.append(mlp_fashion.score(x_noisy_fashion, y_fashion_te_subset))

print("Noise σ | MNIST | Fashion | Diff")
print("-" * 32)
for noise, acc_m, acc_f in zip(noise_levels_comp, acc_mnist_comp, acc_fashion_comp):
    diff = acc_m - acc_f
    print(f"{noise:6.2f} | {acc_m:.3f} | {acc_f:.3f}  | {diff:+.3f}")

# AUC comparative
auc_mnist_comp = np.trapz(acc_mnist_comp, noise_levels_comp)
auc_fashion_comp = np.trapz(acc_fashion_comp, noise_levels_comp)
print(f"\nAUC MNIST: {auc_mnist_comp:.3f}")
print(f"AUC FashionMNIST: {auc_fashion_comp:.3f}")
print(f"Rapporto robustezza: {auc_mnist_comp/auc_fashion_comp:.2f}x MNIST più robusto")

# Analisi top errori FashionMNIST (precedente grafico rimosso)
print(f"\nTOP CONFUSIONI FASHIONMNIST:")
print("-" * 30)

fashion_confusion_pairs = []
for i in range(10):
    for j in range(10):
        if i != j and cm_fashion[i, j] > 0:
            fashion_confusion_pairs.append({
                'true_class': fashion_classes[i],
                'pred_class': fashion_classes[j],
                'true_idx': i,
                'pred_idx': j,
                'count': cm_fashion[i, j]
            })

df_fashion_confusion = pd.DataFrame(fashion_confusion_pairs)
top_5_fashion = df_fashion_confusion.nlargest(5, 'count')

print("Top 5 confusioni più frequenti:")
for _, row in top_5_fashion.iterrows():
    print(f"{row['true_class'][:8]} → {row['pred_class'][:8]}: {row['count']} errori")

# Statistiche comparative dataset
print(f"\nSTATISTICHE COMPARATIVE DATASET:")
print("-" * 35)
print(f"Errori totali MNIST: {total_errors} ({(total_errors/10000)*100:.1f}%)")

fashion_errors = np.sum(y_pred_fashion != fashion_te_labels)
print(f"Errori totali FashionMNIST: {fashion_errors} ({(fashion_errors/10000)*100:.1f}%)")
print(f"Rapporto complessità: {fashion_errors/total_errors:.1f}x più errori su FashionMNIST")


# %% [markdown]
#  ### Discussione finale e conclusioni Punto Bonus
# 
# 
# 
#  **Gap di complessità quantificato:**
# 
# 
# 
#  FashionMNIST dimostra essere significativamente più challenging di MNIST con una perdita di 1.35 punti percentuali (87.63% vs 98.10%), rappresentando un aumento del 54% nel tasso di errore relativo. Questo gap riflette la maggiore complessità intrinseca degli indumenti rispetto alle cifre: forme più variabili, texture, prospettive multiple e sovrapposizioni strutturali.
# 
# 
# 
#  **Pattern di vulnerabilità classe-specifici:**
# 
# 
# 
#  L'analisi per classe rivela vulnerabilità distinctive: 'Shirt' (classe 6) e 'Pullover' (classe 2) mostrano le accuratezze più basse a causa della similitudine morfologica, mentre 'Trouser' (classe 1) e 'Bag' (classe 8) mantengono prestazioni robuste grazie a forme distintive. Le confusioni dominanti (Shirt→T-shirt, Pullover→Coat) riflettono ambiguità semantiche genuine anche per osservatori umani.
# 
# 
# 
#  **Robustezza al rumore comparativa:**
# 
# 
# 
#  FashionMNIST mostra degradazione più rapida al rumore (AUC 1.42) rispetto a MNIST (AUC 1.89), con rapporto di robustezza 1.33x favorevole a MNIST. Questo pattern indica che le features dei capi di abbigliamento sono intrinsecamente più sensibili alle perturbazioni, probabilmente per la maggiore dipendenza da dettagli testurali e geometrici sottili.
# 
# 
# 
#  **Efficacia dell'architettura ottimale:**
# 
# 
# 
#  L'architettura MLP ottimale, progettata su MNIST, mantiene performance competitive su FashionMNIST (87.63%), dimostrando buona generalizzazione cross-domain. Tuttavia, il gap prestazionale suggerisce che architetture più sofisticate (CNN, attention mechanisms) potrebbero fornire benefici maggiori su task visivi complessi rispetto a dataset semplificati come MNIST.
# 
# 
# 
#  **Implicazioni per model selection e deployment:**
# 
# 
# 
#  I risultati evidenziano l'importanza della validazione cross-domain: modelli che eccellono su benchmark semplici (MNIST) possono mostrare limitazioni su applicazioni reali più complesse. Per deployment in domini visuali complessi, si raccomanda validazione su dataset eterogenei e considerazione di architetture specificamente progettate per robustezza alle variazioni intra-classe e complessità morfologica elevata.

# %% [markdown]
#  ## Conclusioni Generali del Progetto
# 
# 
# 
#  ### Riepilogo dei risultati principali
# 
# 
# 
#  **Punto A - Analisi Iperparametri:**
# 
#  - Identificate architetture ottimali: MLP(250n_1S_lr0.001) con 98.10% e CNN(extended_lr0.001) con 98.85%
# 
#  - Learning rate critico: range ottimale 0.001-0.01, collasso catastrofico a 0.1
# 
#  - Profondità controproducente: 1 strato supera 2 strati di +2.2 punti su MNIST
# 
#  - Efficienza dominata da MLP: rapporto 5.3x favorevole per accuratezza/tempo
# 
# 
# 
#  **Punto B - Cifre difficili:**
# 
#  - Gerarchia difficoltà: 8(2.8%), 2(2.5%), 5(2.4%) vs 0(<1%), 1(<1%)
# 
#  - Pattern confusione morfologicamente giustificati: 4→9, 7→2, 8→3
# 
#  - Calibrazione confidenza eccellente: correlazione r=0.78 per early warning
# 
#  - 190 errori totali su 10K (1.90% tasso globale) con distribuzione controllata
# 
# 
# 
#  **Punto C - Robustezza al rumore:**
# 
#  - Soglie operative definite: σ≤0.15(>95%), σ≤0.20(>90%), σ≤0.35(>80%)
# 
#  - Degradazione graduale, non catastrofica: tasso 1.01 acc/σ
# 
#  - Vulnerabilità classe-specifiche: cifra 4 più vulnerabile, cifra 1 più robusta
# 
#  - Allineamento con percezione umana per σ≥0.3
# 
# 
# 
#  **Punto D - Riduzione dati:**
# 
#  - Robustezza a scarsità dati: 10% dati → 94.45% accuratezza (-3.4 punti)
# 
#  - Scaling temporale quasi lineare con benefici efficiency per dataset ridotti
# 
#  - Overfitting controllato anche con dati limitati
# 
#  - Soglie pratiche: 25%(>95%), 10%(>94%), 5%(>92%)
# 
# 
# 
#  **Punto E - Training con rumore:**
# 
#  - Data augmentation efficace: σ=0.15 ottimale con +8.7% AUC miglioramento
# 
#  - Range efficace σ=0.10-0.20 per regolarizzazione benefica
# 
#  - Benefici concentrati su rumore test moderato (σ=0.2-0.3)
# 
#  - Strategia deployment robusto senza modifiche architettoniche
# 
# 
# 
#  **Punto Bonus - FashionMNIST:**
# 
#  - Gap complessità quantificato: -1.35 punti (54% aumento tasso errore)
# 
#  - Robustezza ridotta: rapporto 1.33x MNIST più robusto al rumore
# 
#  - Confusioni semanticamente giustificate tra indumenti simili
# 
#  - Validazione importanza testing cross-domain
# 
# 
# 
#  ### Insights metodologici trasversali
# 
# 
# 
#  **Architettura e iperparametri:**
# 
#  La ricerca sistematica conferma che per task visivi semplici come MNIST, architetture snelle ben calibrate superano configurazioni complesse. Il learning rate emerge come iperparametro più critico, mentre la profondità aggiuntiva introduce overfitting senza benefici su dataset a complessità limitata.
# 
# 
# 
#  **Robustezza e generalizzazione:**
# 
#  I modelli dimostrano resilienza intrinseca a condizioni avverse (rumore, dati limitati) quando l'architettura è appropriata al task. La data augmentation con rumore controllato fornisce miglioramenti significativi "gratuiti" senza costi architettonali.
# 
# 
# 
#  **Efficienza computazionale:**
# 
#  Il trade-off accuratezza-tempo rivela che per molte applicazioni pratiche, configurazioni moderate offrono il miglior valore: MLP(100, lr=0.01) raggiunge 97.3% in <10 secondi, ideale per prototipazione e deployment con vincoli temporali.
# 
# 
# 
#  **Calibrazione e affidabilità:**
# 
#  I modelli mostrano eccellente autoconsapevolezza attraverso calibrazione delle confidenze, fornendo meccanismi naturali di quality control per deployment critico senza modifiche architettoniche aggiuntive.
# 
# 
# 
#  ### Raccomandazioni strategiche per applicazioni reali
# 
# 
# 
#  **Per sviluppo rapido e prototipazione:**
# 
#  - MLP(100, lr=0.01) per iterazione veloce e proof-of-concept
# 
#  - Dataset 10-25% per validazione iniziale con mantenimento >94% prestazioni
# 
#  - Training con rumore σ=0.15 per robustezza immediata
# 
# 
# 
#  **Per deployment critico:**
# 
#  - MLP(250, lr=0.001) per massimo bilanciamento prestazioni-efficienza
# 
#  - Soglie confidenza <0.80 per escalation manuale
# 
#  - Validazione cross-domain obbligatoria prima del deployment
# 
# 
# 
#  **Per massimizzazione prestazioni:**
# 
#  - CNN extended quando costo computazionale giustificabile
# 
#  - Architetture specializzate per domini complessi (FashionMNIST-like)
# 
#  - Data augmentation sistematica per robustezza operativa
# 


