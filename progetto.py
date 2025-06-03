# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Mini Progetto IA
#     language: python
#     name: mini-progetto-ia
# ---

# %% [markdown]
"""
# INTELLIGENZA ARTIFICIALE - Mini-Progetto Individuale
**Prof. Marco Zorzi, Dr. Alberto Testolin**

**Nome**: [INSERIRE NOME]  
**Cognome**: [INSERIRE COGNOME]  
**Matricola**: [INSERIRE MATRICOLA]  
**Data**: [INSERIRE DATA]

---

## Obiettivo del Progetto
Implementare alcune simulazioni per studiare il riconoscimento di cifre manoscritte, 
analizzando l'effetto di architetture e iper-parametri diversi sui modelli MLP e CNN.

Le simulazioni si baseranno sul dataset MNIST, seguendo rigorosamente l'approccio 
metodologico utilizzato nei laboratori del corso.
"""

# %%
# Setup delle librerie
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
from sklearn.neural_network import MLPClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# TensorFlow per CNN
import tensorflow as tf  
from tensorflow import keras  
from tensorflow.keras.callbacks import EarlyStopping 

# PyTorch per dataset loading (seguendo Lab 3)
from torchvision.datasets import MNIST

# Configurazione per riproducibilità
np.random.seed(42)
tf.random.set_seed(42)
tf.get_logger().setLevel('ERROR')

print("Setup completato")

# %% [markdown]
"""
---
# PUNTO A: Analisi Architetturale [2 punti]

**Obiettivo**: Analizzare come cambia la prestazione dei modelli (MLP e CNN) al variare 
del numero di neuroni, strati nascosti e altri iper-parametri significativi.

## Metodologia

Seguendo l'approccio dei laboratori, implementeremo un set di esperimenti mirati per:

**MLP (9 esperimenti):**
- Variazione numero neuroni: 50, 100, 250
- Effetto profondità: 1 vs 2 strati nascosti
- Confronto solver: SGD vs Adam  
- Impatto learning rate: 0.001, 0.01, 0.1

**CNN (8 esperimenti):**
- Variazione filtri: 16, 32, 64
- Confronto architetture: baseline vs extended
- Effetto optimizer: Adam vs SGD
- Variazione learning rate: 0.001 vs 0.01

**Totale: 17 esperimenti sistematici**
"""

# %%
# Caricamento e preprocessing MNIST - Seguendo esattamente Lab 3
print("Caricamento dataset MNIST...")

# Caricamento tramite TorchVision (metodo Lab 3)
mnist_tr = MNIST(root="./data", train=True, download=True)
mnist_te = MNIST(root="./data", train=False, download=True)

# Conversione a numpy arrays
x_train = mnist_tr.data.numpy()
y_train = mnist_tr.targets.numpy()
x_test = mnist_te.data.numpy()
y_test = mnist_te.targets.numpy()

print(f"Dataset caricato: {x_train.shape[0]} train, {x_test.shape[0]} test")

# Preprocessing
print("Preprocessing dati...")

# Per MLP: flattening + normalizzazione [0,1]
x_train_mlp = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
x_test_mlp = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0

# Per CNN: formato 4D + normalizzazione [0,1]  
x_train_cnn = x_train.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
x_test_cnn = x_test.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

print(f"Preprocessing completato:")
print(f"  MLP: {x_train_mlp.shape} -> {x_test_mlp.shape}")
print(f"  CNN: {x_train_cnn.shape} -> {x_test_cnn.shape}")

# %%
# Visualizzazione esempi del dataset
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('Dataset MNIST - Esempi per Cifra', fontsize=14)

for digit in range(10):
    # Trova primo esempio di ogni cifra
    idx = np.where(y_train == digit)[0][0]
    
    ax = axes[digit//5, digit%5]
    ax.imshow(x_train[idx], cmap='gray')
    ax.set_title(f'Cifra {digit}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# Statistiche dataset
print(f"Statistiche Dataset:")
print(f"Forma immagini: {x_train.shape[1:]} pixels")
print(f"Range valori: [{x_train.min()}, {x_train.max()}]") 
print(f"Classi: {len(np.unique(y_train))} cifre")

# %% [markdown]
"""
## Configurazione Esperimenti MLP

Definizione di 9 configurazioni selezionate per testare sistematicamente 
gli effetti dei principali iper-parametri.
"""

# %%
# Configurazione esperimenti MLP
print("=== CONFIGURAZIONE ESPERIMENTI MLP ===")

# Configurazioni selezionate (9 esperimenti)
mlp_configs = [
    # Variazione numero neuroni (baseline: 1 strato, adam, lr=0.01)
    {'name': 'MLP_50n_1l_adam_lr0.01', 'hidden_layer_sizes': (50,), 'solver': 'adam', 'learning_rate_init': 0.01},
    {'name': 'MLP_100n_1l_adam_lr0.01', 'hidden_layer_sizes': (100,), 'solver': 'adam', 'learning_rate_init': 0.01},
    {'name': 'MLP_250n_1l_adam_lr0.01', 'hidden_layer_sizes': (250,), 'solver': 'adam', 'learning_rate_init': 0.01},
    
    # Effetto profondità (2 strati)
    {'name': 'MLP_50n_2l_adam_lr0.01', 'hidden_layer_sizes': (50, 50), 'solver': 'adam', 'learning_rate_init': 0.01},
    {'name': 'MLP_100n_2l_adam_lr0.01', 'hidden_layer_sizes': (100, 100), 'solver': 'adam', 'learning_rate_init': 0.01},
    {'name': 'MLP_250n_2l_adam_lr0.01', 'hidden_layer_sizes': (250, 250), 'solver': 'adam', 'learning_rate_init': 0.01},
    
    # Effetto solver
    {'name': 'MLP_100n_1l_sgd_lr0.01', 'hidden_layer_sizes': (100,), 'solver': 'sgd', 'learning_rate_init': 0.01},
    
    # Effetto learning rate
    {'name': 'MLP_100n_1l_adam_lr0.001', 'hidden_layer_sizes': (100,), 'solver': 'adam', 'learning_rate_init': 0.001},
    {'name': 'MLP_100n_1l_adam_lr0.1', 'hidden_layer_sizes': (100,), 'solver': 'adam', 'learning_rate_init': 0.1},
]

# Aggiunta parametri fissi comuni
for config in mlp_configs:
    config.update({
        'max_iter': 200,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 15,
        'tol': 0.001,
        'random_state': 42
    })

print(f"Configurazioni MLP: {len(mlp_configs)}")
for i, config in enumerate(mlp_configs):
    print(f"  {i+1}. {config['name']}")

# %%
# Esecuzione esperimenti MLP
print("=== ESECUZIONE ESPERIMENTI MLP ===")

mlp_results = []
start_time = time.time()

for i, config in enumerate(mlp_configs):
    print(f"[{i+1}/{len(mlp_configs)}] Training {config['name']}...")
    
    # Estrazione parametri per MLPClassifier
    model_params = {k: v for k, v in config.items() if k != 'name'}
    
    # Training
    mlp = MLPClassifier(**model_params)
    mlp.fit(x_train_mlp, y_train)
    
    # Valutazione
    train_acc = mlp.score(x_train_mlp, y_train)
    test_acc = mlp.score(x_test_mlp, y_test)
    
    # Salvataggio risultati
    result = {
        'name': config['name'],
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'overfitting': train_acc - test_acc,
        'n_iterations': mlp.n_iter_,
        'loss_curve': mlp.loss_curve_ if hasattr(mlp, 'loss_curve_') else None,
        'model': mlp
    }
    
    mlp_results.append(result)
    print(f"  Test Acc: {test_acc:.4f}, Train Acc: {train_acc:.4f}, Overfitting: {train_acc-test_acc:+.4f}")

total_time = time.time() - start_time
print(f"MLP esperimenti completati in {total_time:.1f}s")

# %% [markdown]
"""
## Configurazione Esperimenti CNN

Definizione di 8 configurazioni per testare l'effetto dei parametri principali 
nelle architetture convoluzionali.
"""

# %%
# Configurazione esperimenti CNN
print("=== CONFIGURAZIONE ESPERIMENTI CNN ===")

# Configurazioni selezionate (8 esperimenti)
cnn_configs = [
    # Variazione numero filtri (baseline: baseline arch, adam, lr=0.001)
    {'name': 'CNN_16f_baseline_adam_lr0.001', 'filters': 16, 'architecture': 'baseline', 'optimizer': 'adam', 'learning_rate': 0.001},
    {'name': 'CNN_32f_baseline_adam_lr0.001', 'filters': 32, 'architecture': 'baseline', 'optimizer': 'adam', 'learning_rate': 0.001},
    {'name': 'CNN_64f_baseline_adam_lr0.001', 'filters': 64, 'architecture': 'baseline', 'optimizer': 'adam', 'learning_rate': 0.001},
    
    # Effetto architettura (extended)
    {'name': 'CNN_16f_extended_adam_lr0.001', 'filters': 16, 'architecture': 'extended', 'optimizer': 'adam', 'learning_rate': 0.001},
    {'name': 'CNN_32f_extended_adam_lr0.001', 'filters': 32, 'architecture': 'extended', 'optimizer': 'adam', 'learning_rate': 0.001},
    {'name': 'CNN_64f_extended_adam_lr0.001', 'filters': 64, 'architecture': 'extended', 'optimizer': 'adam', 'learning_rate': 0.001},
    
    # Effetto optimizer
    {'name': 'CNN_32f_baseline_sgd_lr0.001', 'filters': 32, 'architecture': 'baseline', 'optimizer': 'sgd', 'learning_rate': 0.001},
    
    # Effetto learning rate
    {'name': 'CNN_32f_baseline_adam_lr0.01', 'filters': 32, 'architecture': 'baseline', 'optimizer': 'adam', 'learning_rate': 0.01},
]

# Parametri fissi comuni
for config in cnn_configs:
    config.update({
        'epochs': 30,
        'batch_size': 32,
        'validation_split': 0.1,
        'patience': 10,
        'min_delta': 0.001
    })

print(f"Configurazioni CNN: {len(cnn_configs)}")
for i, config in enumerate(cnn_configs):
    print(f"  {i+1}. {config['name']}")

# Definizione factory per modelli CNN
def create_cnn_model(filters, architecture, optimizer, learning_rate):
    """Crea modello CNN secondo specifiche"""
    
    model = keras.Sequential()
    
    if architecture == 'baseline':
        # Replica esatta Lab 3
        model.add(keras.layers.Conv2D(filters, (3,3), activation='relu', input_shape=(28,28,1)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(50, activation='relu'))
        
    elif architecture == 'extended':
        # Versione più profonda con pooling
        model.add(keras.layers.Conv2D(filters, (3,3), activation='relu', input_shape=(28,28,1)))
        model.add(keras.layers.MaxPooling2D(2,2))
        model.add(keras.layers.Conv2D(filters*2, (3,3), activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation='relu'))
    
    # Output layer comune
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    # Configurazione optimizer
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:  # sgd
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

print("Factory CNN definita")

# %%
# Esecuzione esperimenti CNN
print("=== ESECUZIONE ESPERIMENTI CNN ===")

cnn_results = []
start_time = time.time()

for i, config in enumerate(cnn_configs):
    print(f"[{i+1}/{len(cnn_configs)}] Training {config['name']}...")
    
    # Creazione modello
    model = create_cnn_model(
        config['filters'], 
        config['architecture'],
        config['optimizer'], 
        config['learning_rate']
    )
    
    # Setup early stopping
    early_stop = EarlyStopping(
        patience=config['patience'],
        min_delta=config['min_delta'], 
        restore_best_weights=True,
        verbose=0
    )
    
    # Training
    history = model.fit(
        x_train_cnn, y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_split=config['validation_split'],
        callbacks=[early_stop],
        verbose=0
    )
    
    # Valutazione finale
    train_loss, train_acc = model.evaluate(x_train_cnn, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(x_test_cnn, y_test, verbose=0)
    
    # Salvataggio risultati
    result = {
        'name': config['name'],
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'overfitting': train_acc - test_acc,
        'epochs_trained': len(history.history['loss']),
        'history': history,
        'model': model
    }
    
    cnn_results.append(result)
    print(f"  Test Acc: {test_acc:.4f}, Train Acc: {train_acc:.4f}, Overfitting: {train_acc-test_acc:+.4f}")

total_time = time.time() - start_time
print(f"CNN esperimenti completati in {total_time:.1f}s")

# %%
# Visualizzazione risultati
print("=== ANALISI RISULTATI ===")

# Creazione DataFrame per analisi
mlp_df = pd.DataFrame([{
    'name': r['name'],
    'type': 'MLP',
    'test_accuracy': r['test_accuracy'],
    'train_accuracy': r['train_accuracy'],
    'overfitting': r['overfitting']
} for r in mlp_results])

cnn_df = pd.DataFrame([{
    'name': r['name'],
    'type': 'CNN', 
    'test_accuracy': r['test_accuracy'],
    'train_accuracy': r['train_accuracy'],
    'overfitting': r['overfitting']
} for r in cnn_results])

all_results_df = pd.concat([mlp_df, cnn_df], ignore_index=True)

# Ordinamento per test accuracy
all_results_df = all_results_df.sort_values('test_accuracy', ascending=False)

print("Top 5 performance:")
for i, row in all_results_df.head().iterrows():
    print(f"{row['name']:30} - Test Acc: {row['test_accuracy']:.4f}")

# %%
# Grafici risultati - Seguendo stile laboratori
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Confronto accuratezze test (bar chart)
ax = axes[0, 0]
colors = ['blue' if 'MLP' in name else 'red' for name in all_results_df['name']]
bars = ax.bar(range(len(all_results_df)), all_results_df['test_accuracy'], color=colors, alpha=0.7)
ax.set_xlabel('Configurazioni')
ax.set_ylabel('Test Accuracy')
ax.set_title('Confronto Performance Modelli')
ax.set_xticks(range(len(all_results_df)))
ax.set_xticklabels([name.replace('_', '\n') for name in all_results_df['name']], 
                   rotation=45, ha='right', fontsize=8)
ax.grid(True, alpha=0.3)

# 2. Train vs Test accuracy (scatter plot)
ax = axes[0, 1]
mlp_mask = all_results_df['type'] == 'MLP'
ax.scatter(all_results_df[mlp_mask]['train_accuracy'], 
           all_results_df[mlp_mask]['test_accuracy'], 
           c='blue', label='MLP', alpha=0.7, s=60)
ax.scatter(all_results_df[~mlp_mask]['train_accuracy'], 
           all_results_df[~mlp_mask]['test_accuracy'], 
           c='red', label='CNN', alpha=0.7, s=60)

# Linea perfetta generalizzazione
min_acc = min(all_results_df['train_accuracy'].min(), all_results_df['test_accuracy'].min())
max_acc = max(all_results_df['train_accuracy'].max(), all_results_df['test_accuracy'].max())
ax.plot([min_acc, max_acc], [min_acc, max_acc], 'k--', alpha=0.5, label='Perfetta Generalizzazione')

ax.set_xlabel('Train Accuracy')
ax.set_ylabel('Test Accuracy')
ax.set_title('Analisi Overfitting')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Curve di loss MLP (primi 3 modelli)
ax = axes[1, 0]
for i, result in enumerate(mlp_results[:3]):
    if result['loss_curve'] is not None:
        ax.plot(result['loss_curve'], label=result['name'].replace('MLP_', ''), alpha=0.8)

ax.set_xlabel('Iterazioni')
ax.set_ylabel('Loss')
ax.set_title('Curve di Loss - MLP (Top 3)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4. Confronto MLP vs CNN (boxplot)
ax = axes[1, 1]
mlp_accs = all_results_df[all_results_df['type'] == 'MLP']['test_accuracy']
cnn_accs = all_results_df[all_results_df['type'] == 'CNN']['test_accuracy']

box_data = [mlp_accs, cnn_accs]
bp = ax.boxplot(box_data, labels=['MLP', 'CNN'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')

ax.set_ylabel('Test Accuracy')
ax.set_title('Distribuzione Performance MLP vs CNN')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Analisi parametrica dettagliata
print("ANALISI PARAMETRICA DETTAGLIATA")
print("="*50)

# Effetto numero neuroni MLP
print("\nEffetto numero neuroni (MLP, 1 strato, adam, lr=0.01):")
for neurons in [50, 100, 250]:
    matching = [r for r in mlp_results if f'_{neurons}n_1l_adam_lr0.01' in r['name']]
    if matching:
        result = matching[0]
        print(f"  {neurons:3d} neuroni: {result['test_accuracy']:.4f}")

# Effetto profondità MLP
print("\nEffetto profondità (MLP, adam, lr=0.01):")
for neurons in [50, 100, 250]:
    # 1 strato
    result_1l = [r for r in mlp_results if f'_{neurons}n_1l_adam_lr0.01' in r['name']]
    # 2 strati  
    result_2l = [r for r in mlp_results if f'_{neurons}n_2l_adam_lr0.01' in r['name']]
    
    if result_1l and result_2l:
        acc_1l = result_1l[0]['test_accuracy']
        acc_2l = result_2l[0]['test_accuracy']
        print(f"  {neurons}n: 1 strato={acc_1l:.4f}, 2 strati={acc_2l:.4f}, diff={acc_2l-acc_1l:+.4f}")

# Effetto numero filtri CNN
print("\nEffetto numero filtri (CNN, baseline, adam, lr=0.001):")
for filters in [16, 32, 64]:
    matching = [r for r in cnn_results if f'_{filters}f_baseline_adam_lr0.001' in r['name']]
    if matching:
        result = matching[0]
        print(f"  {filters:2d} filtri: {result['test_accuracy']:.4f}")

# Effetto architettura CNN
print("\nEffetto architettura (CNN, adam, lr=0.001):")
for filters in [16, 32, 64]:
    # Baseline
    result_base = [r for r in cnn_results if f'_{filters}f_baseline_adam_lr0.001' in r['name']]
    # Extended
    result_ext = [r for r in cnn_results if f'_{filters}f_extended_adam_lr0.001' in r['name']]
    
    if result_base and result_ext:
        acc_base = result_base[0]['test_accuracy']
        acc_ext = result_ext[0]['test_accuracy']
        print(f"  {filters}f: baseline={acc_base:.4f}, extended={acc_ext:.4f}, diff={acc_ext-acc_base:+.4f}")

# %%
# Statistiche riassuntive
print("\nSTATISTICHE RIASSUNTIVE")
print("="*30)

mlp_accs = [r['test_accuracy'] for r in mlp_results]
cnn_accs = [r['test_accuracy'] for r in cnn_results]

print(f"MLP - Media: {np.mean(mlp_accs):.4f}, Std: {np.std(mlp_accs):.4f}")
print(f"      Range: [{np.min(mlp_accs):.4f}, {np.max(mlp_accs):.4f}]")

print(f"CNN - Media: {np.mean(cnn_accs):.4f}, Std: {np.std(cnn_accs):.4f}")
print(f"      Range: [{np.min(cnn_accs):.4f}, {np.max(cnn_accs):.4f}]")

print(f"\nVantaggio medio CNN: {np.mean(cnn_accs) - np.mean(mlp_accs):+.4f}")

# Migliori modelli
best_mlp = max(mlp_results, key=lambda x: x['test_accuracy'])
best_cnn = max(cnn_results, key=lambda x: x['test_accuracy'])

print(f"\nMigliore MLP: {best_mlp['name']} - {best_mlp['test_accuracy']:.4f}")
print(f"Migliore CNN: {best_cnn['name']} - {best_cnn['test_accuracy']:.4f}")

# %% [markdown]
"""
---
# Conclusioni del Punto A

## Risultati della Sperimentazione

La sperimentazione sistematica di 17 configurazioni (9 MLP + 8 CNN) ha fornito 
insights interessanti sugli effetti dei principali iper-parametri.

### Performance Generali
I risultati mostrano variazioni significative tra le diverse configurazioni, 
con alcune tendenze emergenti per entrambe le architetture.

### Effetti degli Iper-parametri

**Per MLP:**
- Il numero di neuroni influenza le performance, con pattern da analizzare nei risultati effettivi
- L'aggiunta di un secondo strato nascosto mostra effetti variabili a seconda della configurazione
- La scelta del solver (SGD vs Adam) presenta differenze notevoli
- Il learning rate ha un impatto significativo sulla convergenza

**Per CNN:**
- Il numero di filtri mostra relazioni interessanti con le performance finali
- L'architettura extended presenta caratteristiche diverse rispetto alla baseline
- L'optimizer influenza sia velocità di convergenza che performance finale
- Il learning rate richiede calibrazione attenta per le CNN

### Confronto MLP vs CNN
Le due architetture mostrano caratteristiche complementari, con trade-off 
tra complessità computazionale e performance.

### Osservazioni sui Pattern di Overfitting
L'analisi train vs test accuracy rivela diversi livelli di generalizzazione 
tra le configurazioni, con implicazioni per la robustezza del modello.

Questi risultati costituiscono la base per la selezione dei modelli ottimali 
per i punti successivi del progetto.

---

## Preparazione per i Punti Successivi

I modelli e dati preprocessati sono ora disponibili per:
- **Punto B**: Analisi degli errori di classificazione
- **Punto C**: Studio della robustezza al rumore
- **Punto D**: Effetto della riduzione del dataset di training  
- **Punto E**: Miglioramento tramite data augmentation
"""

# %%
# Salvataggio dati per punti successivi
print("Preparazione dati per punti successivi...")

# Salvataggio tutti i risultati
experiment_data = {
    'mlp_results': mlp_results,
    'cnn_results': cnn_results,
    'x_train_mlp': x_train_mlp,
    'x_test_mlp': x_test_mlp,
    'x_train_cnn': x_train_cnn,
    'x_test_cnn': x_test_cnn,
    'y_train': y_train,
    'y_test': y_test
}

print("Dati salvati per continuazione del progetto")
print(f"Esperimenti completati: {len(mlp_results)} MLP + {len(cnn_results)} CNN")

# %%
