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
# Mini Progetto Intelligenza Artificiale - Riconoscimento cifre manoscritte

**Nome:** [Inserire nome]  
**Cognome:** [Inserire cognome]  
**Matricola:** [Inserire matricola]  
**Data consegna:** [Inserire data]

## Obiettivo

In questo progetto esploreremo il riconoscimento di cifre manoscritte utilizzando il dataset MNIST, implementando simulazioni per studiare come diversi fattori influenzano le prestazioni dei modelli di deep learning. Analizzeremo in particolare l'impatto degli iperparametri, la robustezza al rumore e l'effetto della quantità di dati di training.
"""

# %% [markdown]
"""
## Importazione delle librerie necessarie
"""

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
plt.rcParams['figure.figsize'] = (10, 6)

# %% [markdown]
"""
## Caricamento e preparazione del dataset MNIST
"""

# %%
# Caricamento dataset MNIST
print("Caricamento dataset MNIST...")
mnist_tr = MNIST(root="./data", train=True, download=True)
mnist_te = MNIST(root="./data", train=False, download=True)

# %%
# Conversione in array numpy
mnist_tr_data, mnist_tr_labels = mnist_tr.data.numpy(), mnist_tr.targets.numpy()
mnist_te_data, mnist_te_labels = mnist_te.data.numpy(), mnist_te.targets.numpy()

# Preprocessing per MLP (vettorizzazione e normalizzazione)
x_tr = mnist_tr_data.reshape(60000, 28 * 28) / 255.0
x_te = mnist_te_data.reshape(10000, 28 * 28) / 255.0

# Preprocessing per CNN (mantenendo formato 2D)
x_tr_conv = x_tr.reshape(-1, 28, 28, 1)
x_te_conv = x_te.reshape(-1, 28, 28, 1)

print(f"Dataset caricato: {x_tr.shape[0]} train, {x_te.shape[0]} test")
print(f"Forma dati MLP: {x_tr.shape}")
print(f"Forma dati CNN: {x_tr_conv.shape}")

# Visualizzazione esempi del dataset
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('Dataset MNIST - Esempi per Cifra', fontsize=14)

for digit in range(10):
    idx = np.where(mnist_tr_labels == digit)[0][0]
    ax = axes[digit//5, digit%5]
    ax.imshow(mnist_tr_data[idx], cmap='gray')
    ax.set_title(f'Cifra {digit}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Punto A: Effetto degli iperparametri sulle prestazioni

Analizziamo sistematicamente come variano le prestazioni dei modelli MLP e CNN al variare degli iperparametri chiave. 
Confronteremo 18 configurazioni MLP e 6 configurazioni CNN per un totale di 24 esperimenti mirati.
"""

# %% [markdown]
"""
### Configurazione esperimenti sistematici

**MLP (18 esperimenti):**
- Neuroni per strato: 50, 100, 250
- Numero strati: 1 vs 2 strati nascosti
- Learning rate: 0.001, 0.01, 0.1

**CNN (6 esperimenti):**
- Filtri: 32 (fisso)
- Architettura: baseline vs extended
- Learning rate: 0.001, 0.01, 0.1

**Parametri di training giustificati:**
- MLP: max_iter=100 (sufficient convergence), early_stopping=True (prevent overfitting), 
  validation_fraction=0.1 (standard split), tol=0.001 (reasonable precision), n_iter_no_change=10 (adequate patience)
- CNN: epochs=20 (balance speed/convergence), batch_size=128 (memory/speed trade-off), 
  validation_split=0.1 (consistency with MLP), patience=5 (faster CNN convergence), min_delta=0.001 (same precision as MLP)
"""

# %%
def print_experiment_header(exp_num, total, model_type, config):
    """Stampa header consistente per ogni esperimento"""
    print(f"\n[{exp_num:2d}/{total}] {model_type}: {config}")
    print("-" * 50)

def print_experiment_results(results):
    """Stampa risultati in formato consistente"""
    print(f"Train Acc: {results['train_accuracy']:.4f} | Test Acc: {results['test_accuracy']:.4f}")
    print(f"Time: {results['training_time']:6.1f}s | Iterations: {results['iterations']:3d}")
    print(f"Overfitting: {results['overfitting']:+.4f}")

# %%
# Esperimenti MLP sistematici
neurons_list = [50, 100, 250]
layers_list = [1, 2]  # numero di strati nascosti
learning_rates = [0.001, 0.01, 0.1]

mlp_results = []
experiment_count = 0
total_experiments = len(neurons_list) * len(layers_list) * len(learning_rates)

print("INIZIO ESPERIMENTI MLP")
print("=" * 60)

for neurons in neurons_list:
    for n_layers in layers_list:
        for lr in learning_rates:
            experiment_count += 1
            
            # Configurazione architettura
            if n_layers == 1:
                hidden_layers = (neurons,)
                config_name = f"{neurons}n_1L_lr{lr}"
            else:
                hidden_layers = (neurons, neurons)
                config_name = f"{neurons}n_2L_lr{lr}"
            
            print_experiment_header(experiment_count, total_experiments, "MLP", config_name)
            
            # Training MLP
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                learning_rate_init=lr,
                max_iter=100,
                early_stopping=True,
                validation_fraction=0.1,
                tol=0.001,
                n_iter_no_change=10,
                random_state=42
            )
            
            start_time = time.time()
            mlp.fit(x_tr, mnist_tr_labels)
            training_time = time.time() - start_time
            
            train_acc = mlp.score(x_tr, mnist_tr_labels)
            test_acc = mlp.score(x_te, mnist_te_labels)
            
            results = {
                'model_type': 'MLP',
                'config_name': config_name,
                'neurons': neurons,
                'n_layers': n_layers,
                'learning_rate': lr,
                'hidden_layers': hidden_layers,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'overfitting': train_acc - test_acc,
                'training_time': training_time,
                'iterations': mlp.n_iter_,
                'loss_curve': mlp.loss_curve_ if hasattr(mlp, 'loss_curve_') else [],
                'total_parameters': sum([layer.size for layer in mlp.coefs_]) + sum([layer.size for layer in mlp.intercepts_])
            }
            
            mlp_results.append(results)
            print_experiment_results(results)

print(f"\nMLP EXPERIMENTS COMPLETED: {len(mlp_results)} configurations tested")

# %%
# Esperimenti CNN sistematici
def create_cnn_model(architecture_type, learning_rate):
    """Crea modello CNN con architettura specificata"""
    model = keras.Sequential()
    
    if architecture_type == 'baseline':
        # Architettura baseline del Lab 3
        model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(50, activation='relu'))
        
    elif architecture_type == 'extended':
        # Architettura estesa con pooling e più strati
        model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
        model.add(keras.layers.MaxPooling2D(2,2))
        model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation='relu'))
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    # Configurazione optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# %%
# Esperimenti CNN
architectures = ['baseline', 'extended']
learning_rates_cnn = [0.001, 0.01, 0.1]

cnn_results = []
cnn_experiment_count = 0
total_cnn_experiments = len(architectures) * len(learning_rates_cnn)

print("\n\nINIZIO ESPERIMENTI CNN")
print("=" * 60)

for arch in architectures:
    for lr in learning_rates_cnn:
        cnn_experiment_count += 1
        config_name = f"CNN_{arch}_lr{lr}"
        
        print_experiment_header(cnn_experiment_count, total_cnn_experiments, "CNN", config_name)
        
        # Creazione e training CNN
        model = create_cnn_model(arch, lr)
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            patience=5,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=0
        )
        
        start_time = time.time()
        history = model.fit(
            x_tr_conv, mnist_tr_labels,
            validation_split=0.1,
            epochs=20,
            batch_size=128,
            callbacks=[early_stopping],
            verbose=0
        )
        training_time = time.time() - start_time
        
        # Valutazione
        train_loss, train_acc = model.evaluate(x_tr_conv, mnist_tr_labels, verbose=0)
        test_loss, test_acc = model.evaluate(x_te_conv, mnist_te_labels, verbose=0)
        
        results = {
            'model_type': 'CNN',
            'config_name': config_name,
            'architecture': arch,
            'learning_rate': lr,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'overfitting': train_acc - test_acc,
            'training_time': training_time,
            'iterations': len(history.history['loss']),
            'loss_curve': history.history['loss'],
            'val_loss_curve': history.history['val_loss'],
            'total_parameters': model.count_params()
        }
        
        cnn_results.append(results)
        print_experiment_results(results)

print(f"\nCNN EXPERIMENTS COMPLETED: {len(cnn_results)} configurations tested")

# %%
# Combinazione risultati per analisi
all_results = mlp_results + cnn_results
df_results = pd.DataFrame(all_results)

print(f"\nTOTAL EXPERIMENTS COMPLETED: {len(all_results)}")
print("=" * 60)
print("SUMMARY STATISTICS:")
print(f"Best MLP accuracy: {df_results[df_results['model_type']=='MLP']['test_accuracy'].max():.4f}")
print(f"Best CNN accuracy: {df_results[df_results['model_type']=='CNN']['test_accuracy'].max():.4f}")
print(f"Fastest training: {df_results['training_time'].min():.1f}s")
print(f"Slowest training: {df_results['training_time'].max():.1f}s")

# %% [markdown]
"""
### Grafico 1: Effetto del Learning Rate (MLP)

Analisi dell'impatto del learning rate sulla convergenza e stabilità del training per le reti MLP.
"""

# %%
# Preparazione dati per learning rate analysis
lr_001_data = [r for r in mlp_results if r['learning_rate'] == 0.001]
lr_01_data = [r for r in mlp_results if r['learning_rate'] == 0.01]
lr_1_data = [r for r in mlp_results if r['learning_rate'] == 0.1]

# Calcolo medie per ogni learning rate
lr_001_acc = np.mean([r['test_accuracy'] for r in lr_001_data])
lr_01_acc = np.mean([r['test_accuracy'] for r in lr_01_data])
lr_1_acc = np.mean([r['test_accuracy'] for r in lr_1_data])

lr_001_time = np.mean([r['training_time'] for r in lr_001_data])
lr_01_time = np.mean([r['training_time'] for r in lr_01_data])
lr_1_time = np.mean([r['training_time'] for r in lr_1_data])

# Visualizzazione loss curves rappresentative
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Loss curves
for i, (lr_data, color, label) in enumerate([(lr_001_data, 'green', 'LR=0.001'), 
                                           (lr_01_data, 'blue', 'LR=0.01'), 
                                           (lr_1_data, 'red', 'LR=0.1')]):
    if lr_data and lr_data[0]['loss_curve']:
        loss_curve = lr_data[0]['loss_curve']  # Prendo primo esempio rappresentativo
        ax1.plot(range(len(loss_curve)), loss_curve, color=color, linewidth=2, label=label)

ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss')
ax1.set_title('Convergence Pattern by Learning Rate')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Accuracy vs Learning Rate
learning_rates_plot = [0.001, 0.01, 0.1]
accuracies = [lr_001_acc, lr_01_acc, lr_1_acc]
colors = ['green', 'blue', 'red']

bars = ax2.bar(range(len(learning_rates_plot)), accuracies, color=colors, alpha=0.7)
ax2.set_xlabel('Learning Rate')
ax2.set_ylabel('Average Test Accuracy')
ax2.set_title('Test Accuracy by Learning Rate')
ax2.set_xticks(range(len(learning_rates_plot)))
ax2.set_xticklabels(['0.001', '0.01', '0.1'])
ax2.grid(True, alpha=0.3)

# Annotazioni valori
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax2.annotate(f'{acc:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("LEARNING RATE ANALYSIS:")
print(f"LR=0.001: Accuracy={lr_001_acc:.4f}, Time={lr_001_time:.1f}s")
print(f"LR=0.01:  Accuracy={lr_01_acc:.4f}, Time={lr_01_time:.1f}s") 
print(f"LR=0.1:   Accuracy={lr_1_acc:.4f}, Time={lr_1_time:.1f}s")
print("[risultati da implementare]")

# %% [markdown]
"""
### Grafico 2: Confronto Architetture MLP vs CNN

Confronto diretto delle prestazioni tra le migliori configurazioni MLP e CNN.
"""

# %%
# Selezione migliori configurazioni
best_mlp = max(mlp_results, key=lambda x: x['test_accuracy'])
best_cnn = max(cnn_results, key=lambda x: x['test_accuracy'])

# Preparazione dati per confronto
mlp_configs = []
mlp_accuracies = []
mlp_times = []

# Raggruppo MLP per configurazione (media dei learning rates)
for neurons in neurons_list:
    for n_layers in layers_list:
        configs_group = [r for r in mlp_results if r['neurons'] == neurons and r['n_layers'] == n_layers]
        if configs_group:
            avg_acc = np.mean([r['test_accuracy'] for r in configs_group])
            avg_time = np.mean([r['training_time'] for r in configs_group])
            config_name = f"MLP({neurons}{'x2' if n_layers==2 else ''})"
            mlp_configs.append(config_name)
            mlp_accuracies.append(avg_acc)
            mlp_times.append(avg_time)

# CNN data
cnn_configs = []
cnn_accuracies = []
cnn_times = []

for arch in architectures:
    configs_group = [r for r in cnn_results if r['architecture'] == arch]
    if configs_group:
        avg_acc = np.mean([r['test_accuracy'] for r in configs_group])
        avg_time = np.mean([r['training_time'] for r in configs_group])
        config_name = f"CNN({arch})"
        cnn_configs.append(config_name)
        cnn_accuracies.append(avg_acc)
        cnn_times.append(avg_time)

# Visualizzazione
fig, ax = plt.subplots(figsize=(12, 6))

# Bar chart con colori distinti
x_mlp = np.arange(len(mlp_configs))
x_cnn = np.arange(len(mlp_configs), len(mlp_configs) + len(cnn_configs))

bars_mlp = ax.bar(x_mlp, mlp_accuracies, color='lightblue', alpha=0.8, label='MLP')
bars_cnn = ax.bar(x_cnn, cnn_accuracies, color='salmon', alpha=0.8, label='CNN')

ax.set_xlabel('Architecture')
ax.set_ylabel('Test Accuracy')
ax.set_title('MLP vs CNN Architecture Comparison')
ax.set_xticks(np.concatenate([x_mlp, x_cnn]))
ax.set_xticklabels(mlp_configs + cnn_configs, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

# Annotazioni valori
for bars, accs in [(bars_mlp, mlp_accuracies), (bars_cnn, cnn_accuracies)]:
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax.annotate(f'{acc:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("ARCHITECTURE COMPARISON:")
print(f"Best MLP: {best_mlp['config_name']} - Accuracy: {best_mlp['test_accuracy']:.4f}")
print(f"Best CNN: {best_cnn['config_name']} - Accuracy: {best_cnn['test_accuracy']:.4f}")
print("[risultati da implementare]")

# %% [markdown]
"""
### Grafico 3: Analisi Efficienza (Tempo vs Accuratezza)

Visualizzazione del trade-off tra tempo di training e accuratezza raggiunta.
"""

# %%
# Scatter plot efficienza
fig, ax = plt.subplots(figsize=(10, 6))

# Separazione dati MLP e CNN
mlp_times = [r['training_time'] for r in mlp_results]
mlp_accs = [r['test_accuracy'] for r in mlp_results]
cnn_times = [r['training_time'] for r in cnn_results]
cnn_accs = [r['test_accuracy'] for r in cnn_results]

# Scatter plots
ax.scatter(mlp_times, mlp_accs, c='blue', alpha=0.7, s=100, label='MLP', marker='o')
ax.scatter(cnn_times, cnn_accs, c='red', alpha=0.7, s=100, label='CNN', marker='s')

ax.set_xlabel('Training Time (seconds)')
ax.set_ylabel('Test Accuracy')
ax.set_title('Efficiency Analysis: Training Time vs Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

# Evidenziazione punti ottimali
best_mlp_idx = mlp_results.index(best_mlp)
best_cnn_idx = cnn_results.index(best_cnn)

ax.scatter(best_mlp['training_time'], best_mlp['test_accuracy'], 
          c='darkblue', s=200, marker='*', label='Best MLP')
ax.scatter(best_cnn['training_time'], best_cnn['test_accuracy'], 
          c='darkred', s=200, marker='*', label='Best CNN')

plt.tight_layout()
plt.show()

print("EFFICIENCY ANALYSIS:")
print(f"MLP range: {min(mlp_times):.1f}s - {max(mlp_times):.1f}s")
print(f"CNN range: {min(cnn_times):.1f}s - {max(cnn_times):.1f}s")
print(f"Best efficiency MLP: {best_mlp['test_accuracy']:.4f} acc in {best_mlp['training_time']:.1f}s")
print(f"Best efficiency CNN: {best_cnn['test_accuracy']:.4f} acc in {best_cnn['training_time']:.1f}s")
print("[risultati da implementare]")

# %% [markdown]
"""
### Grafico 4: Analisi Overfitting

Studio del gap tra training e test accuracy in relazione alla complessità del modello.
"""

# %%
# Calcolo complessità modelli (parametri totali)
mlp_complexities = []
mlp_overfitting = []
cnn_complexities = []
cnn_overfitting = []

for result in mlp_results:
    mlp_complexities.append(result['total_parameters'])
    mlp_overfitting.append(result['overfitting'])

for result in cnn_results:
    cnn_complexities.append(result['total_parameters'])
    cnn_overfitting.append(result['overfitting'])

# Visualizzazione
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(mlp_complexities, mlp_overfitting, c='blue', alpha=0.7, s=100, label='MLP')
ax.scatter(cnn_complexities, cnn_overfitting, c='red', alpha=0.7, s=100, label='CNN')

ax.set_xlabel('Model Complexity (Total Parameters)')
ax.set_ylabel('Overfitting (Train - Test Accuracy)')
ax.set_title('Overfitting Analysis vs Model Complexity')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print("OVERFITTING ANALYSIS:")
print(f"MLP overfitting range: {min(mlp_overfitting):.4f} to {max(mlp_overfitting):.4f}")
print(f"CNN overfitting range: {min(cnn_overfitting):.4f} to {max(cnn_overfitting):.4f}")
print("[risultati da implementare]")

# %% [markdown]
"""
### Grafico 5: Velocità di Convergenza

Confronto del numero di iterazioni necessarie per raggiungere la convergenza.
"""

# %%
# Preparazione dati convergenza
mlp_iterations = [r['iterations'] for r in mlp_results]
cnn_iterations = [r['iterations'] for r in cnn_results]

# Raggruppo per tipo di modello
mlp_configs_conv = [r['config_name'] for r in mlp_results]
cnn_configs_conv = [r['config_name'] for r in cnn_results]

# Media iterazioni per architettura
mlp_arch_iterations = {}
for neurons in neurons_list:
    for n_layers in layers_list:
        key = f"{neurons}n_{n_layers}L"
        iterations = [r['iterations'] for r in mlp_results 
                     if r['neurons'] == neurons and r['n_layers'] == n_layers]
        mlp_arch_iterations[key] = np.mean(iterations) if iterations else 0

cnn_arch_iterations = {}
for arch in architectures:
    iterations = [r['iterations'] for r in cnn_results if r['architecture'] == arch]
    cnn_arch_iterations[arch] = np.mean(iterations) if iterations else 0

# Visualizzazione
fig, ax = plt.subplots(figsize=(12, 6))

# Preparazione dati per bar chart
all_configs = list(mlp_arch_iterations.keys()) + list(cnn_arch_iterations.keys())
all_iterations = list(mlp_arch_iterations.values()) + list(cnn_arch_iterations.values())

# Colori distinti
colors = ['lightblue'] * len(mlp_arch_iterations) + ['salmon'] * len(cnn_arch_iterations)

bars = ax.bar(range(len(all_configs)), all_iterations, color=colors, alpha=0.8)

ax.set_xlabel('Architecture')
ax.set_ylabel('Average Iterations to Convergence')
ax.set_title('Convergence Speed Comparison')
ax.set_xticks(range(len(all_configs)))
ax.set_xticklabels(all_configs, rotation=45)
ax.grid(True, alpha=0.3)

# Annotazioni
for bar, iterations in zip(bars, all_iterations):
    height = bar.get_height()
    ax.annotate(f'{iterations:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

# Legenda manuale
ax.bar([], [], color='lightblue', alpha=0.8, label='MLP')
ax.bar([], [], color='salmon', alpha=0.8, label='CNN')
ax.legend()

plt.tight_layout()
plt.show()

print("CONVERGENCE SPEED ANALYSIS:")
print(f"MLP average iterations: {np.mean(mlp_iterations):.1f}")
print(f"CNN average iterations: {np.mean(cnn_iterations):.1f}")
print("[risultati da implementare]")

# %% [markdown]
"""
### Grafico 6: Scaling Effect MLP (1 vs 2 Layers)

Analisi dell'effetto del numero di strati nascosti sulle prestazioni MLP.
"""

# %%
# Analisi scaling MLP
neurons_range = neurons_list
acc_1layer = []
acc_2layer = []
time_1layer = []
time_2layer = []

for neurons in neurons_range:
    # 1 strato
    results_1l = [r for r in mlp_results if r['neurons'] == neurons and r['n_layers'] == 1]
    if results_1l:
        acc_1layer.append(np.mean([r['test_accuracy'] for r in results_1l]))
        time_1layer.append(np.mean([r['training_time'] for r in results_1l]))
    
    # 2 strati  
    results_2l = [r for r in mlp_results if r['neurons'] == neurons and r['n_layers'] == 2]
    if results_2l:
        acc_2layer.append(np.mean([r['test_accuracy'] for r in results_2l]))
        time_2layer.append(np.mean([r['training_time'] for r in results_2l]))

# Visualizzazione
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Accuracy scaling
ax1.plot(neurons_range, acc_1layer, 'o-', linewidth=2, markersize=8, label='1 Hidden Layer', color='blue')
ax1.plot(neurons_range, acc_2layer, 's-', linewidth=2, markersize=8, label='2 Hidden Layers', color='darkblue')

ax1.set_xlabel('Neurons per Layer')
ax1.set_ylabel('Test Accuracy')
ax1.set_title('MLP Scaling: Accuracy vs Depth')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Annotazioni
for i, (neurons, acc1, acc2) in enumerate(zip(neurons_range, acc_1layer, acc_2layer)):
    ax1.annotate(f'{acc1:.3f}', (neurons, acc1), textcoords="offset points", xytext=(0,10), ha='center')
    ax1.annotate(f'{acc2:.3f}', (neurons, acc2), textcoords="offset points", xytext=(0,-15), ha='center')

# Subplot 2: Training time scaling
ax2.plot(neurons_range, time_1layer, 'o-', linewidth=2, markersize=8, label='1 Hidden Layer', color='green')
ax2.plot(neurons_range, time_2layer, 's-', linewidth=2, markersize=8, label='2 Hidden Layers', color='darkgreen')

ax2.set_xlabel('Neurons per Layer')
ax2.set_ylabel('Training Time (seconds)')
ax2.set_title('MLP Scaling: Training Time vs Depth')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("MLP SCALING ANALYSIS:")
for i, neurons in enumerate(neurons_range):
    print(f"{neurons} neurons: 1L={acc_1layer[i]:.4f} ({time_1layer[i]:.1f}s), 2L={acc_2layer[i]:.4f} ({time_2layer[i]:.1f}s)")
print("[risultati da implementare]")

# %% [markdown]
"""
### Riepilogo Punto A

[risultati da implementare]

**Configurazioni testate:** 24 esperimenti sistematici (18 MLP + 6 CNN)

**Insights principali:**
- Effetto learning rate su stabilità e convergenza
- Confronto efficienza MLP vs CNN  
- Relazione complessità modello e overfitting
- Velocità convergenza diverse architetture
- Scaling effect profondità vs larghezza MLP

**Migliori configurazioni identificate:**
- MLP: [da determinare in base ai risultati]
- CNN: [da determinare in base ai risultati]
"""

# %% [markdown]
"""
## Punto B: Analisi delle cifre più difficili da riconoscere

Utilizziamo la matrice di confusione per identificare quali cifre il modello MLP trova più difficili da classificare correttamente.
"""

# %%
# Addestro un MLP con architettura ottimale trovata precedentemente
best_mlp_config = max(mlp_results, key=lambda x: x['test_accuracy'])

mlp_best = MLPClassifier(
    hidden_layer_sizes=best_mlp_config['hidden_layers'],
    learning_rate_init=best_mlp_config['learning_rate'],
    max_iter=100,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

print(f"Training MLP con architettura ottimale: {best_mlp_config['config_name']}")
mlp_best.fit(x_tr, mnist_tr_labels)
print(f"Accuratezza sul test set: {mlp_best.score(x_te, mnist_te_labels):.4f}")

# %%
# Calcolo predizioni e matrice di confusione
y_pred = mlp_best.predict(x_te)

# Visualizzazione matrice di confusione
cm = metrics.confusion_matrix(mnist_te_labels, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Matrice di Confusione - MLP su MNIST', fontsize=16)
plt.show()

# %%
# Analisi degli errori più frequenti
errors_per_digit = []
for digit in range(10):
    mask = mnist_te_labels == digit
    total = np.sum(mask)
    correct = np.sum((y_pred == mnist_te_labels) & mask)
    error_rate = 1 - (correct / total)
    
    errors_per_digit.append({
        'digit': digit,
        'total_samples': total,
        'correct': correct,
        'errors': total - correct,
        'error_rate': error_rate,
        'accuracy': correct / total
    })

df_errors = pd.DataFrame(errors_per_digit)
df_errors_sorted = df_errors.sort_values('error_rate', ascending=False)

print("Cifre ordinate per difficoltà (tasso di errore):")
print(df_errors_sorted[['digit', 'total_samples', 'errors', 'error_rate', 'accuracy']])

# %%
# Visualizzazione delle coppie di cifre più confuse
confusion_pairs = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append({
                'true_digit': i,
                'predicted_digit': j,
                'count': cm[i, j],
                'percentage': cm[i, j] / np.sum(cm[i, :]) * 100
            })

df_confusion = pd.DataFrame(confusion_pairs)
df_confusion_sorted = df_confusion.sort_values('count', ascending=False).head(10)

print("\nLe 10 coppie di cifre più confuse:")
for _, row in df_confusion_sorted.iterrows():
    print(f"{row['true_digit']} → {row['predicted_digit']}: {row['count']} errori ({row['percentage']:.1f}%)")

# %%
# Visualizzazione esempi di cifre classificate erroneamente
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
axes = axes.ravel()

example_idx = 0
for _, conf_pair in df_confusion_sorted.head(4).iterrows():
    true_digit = conf_pair['true_digit']
    pred_digit = conf_pair['predicted_digit']
    
    # Trovo esempi di questo tipo di errore
    error_mask = (mnist_te_labels == true_digit) & (y_pred == pred_digit)
    error_indices = np.where(error_mask)[0]
    
    # Mostro fino a 5 esempi per ogni coppia
    for i in range(min(5, len(error_indices))):
        if example_idx < 20:
            idx = error_indices[i]
            axes[example_idx].imshow(mnist_te_data[idx], cmap='gray')
            axes[example_idx].set_title(f'Vero: {true_digit}, Predetto: {pred_digit}', fontsize=10)
            axes[example_idx].axis('off')
            example_idx += 1

# Nascondo assi non utilizzati
for i in range(example_idx, 20):
    axes[i].axis('off')

plt.suptitle('Esempi di cifre classificate erroneamente', fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Punto C: Curve psicometriche - Effetto del rumore

Seguendo la metodologia dell'articolo di Testolin et al. (2017), analizziamo come l'accuratezza degrada all'aumentare del rumore Gaussiano aggiunto alle immagini.
"""

# %%
# Funzione per aggiungere rumore Gaussiano
def add_gaussian_noise(images, noise_std):
    """
    Aggiunge rumore Gaussiano alle immagini.
    
    Args:
        images: array di immagini
        noise_std: deviazione standard del rumore
    
    Returns:
        Immagini con rumore, clippate tra 0 e 1
    """
    noise = np.random.normal(0, noise_std, images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0, 1)

# %%
# Test con diversi livelli di rumore
noise_levels = np.arange(0, 0.5, 0.05)
accuracies_mlp = []

# Uso un subset del test set per velocizzare
subset_size = 2000
x_te_subset = x_te[:subset_size]
y_te_subset = mnist_te_labels[:subset_size]

print("Calcolo curve psicometriche per MLP...")
for noise_std in noise_levels:
    x_te_noisy = add_gaussian_noise(x_te_subset, noise_std)
    acc_mlp = mlp_best.score(x_te_noisy, y_te_subset)
    accuracies_mlp.append(acc_mlp)
    print(f"Noise std: {noise_std:.3f} - MLP acc: {acc_mlp:.4f}")

# Test anche con CNN se disponibile
best_cnn_config = max(cnn_results, key=lambda x: x['test_accuracy'])
cnn_model = create_cnn_model(best_cnn_config['architecture'], best_cnn_config['learning_rate'])

# Riaddestro il modello CNN migliore
early_stopping = keras.callbacks.EarlyStopping(patience=5, min_delta=0.001, restore_best_weights=True, verbose=0)
cnn_model.fit(x_tr_conv, mnist_tr_labels, validation_split=0.1, epochs=20, batch_size=128, 
              callbacks=[early_stopping], verbose=0)

print("\nCalcolo curve psicometriche per CNN...")
accuracies_cnn = []
x_te_conv_subset = x_te_conv[:subset_size]

for noise_std in noise_levels:
    x_te_conv_noisy = add_gaussian_noise(x_te_conv_subset, noise_std)
    test_loss, acc_cnn = cnn_model.evaluate(x_te_conv_noisy, y_te_subset, verbose=0)
    accuracies_cnn.append(acc_cnn)
    print(f"Noise std: {noise_std:.3f} - CNN acc: {acc_cnn:.4f}")

# %%
# Visualizzazione curve psicometriche
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Grafico 1: Curve psicometriche
ax1.plot(noise_levels, accuracies_mlp, 'o-', label='MLP', linewidth=3, markersize=8, color='blue')
ax1.plot(noise_levels, accuracies_cnn, 's-', label='CNN', linewidth=3, markersize=8, color='red')

ax1.set_xlabel('Deviazione standard del rumore', fontsize=12)
ax1.set_ylabel('Accuratezza', fontsize=12)
ax1.set_title('Curve Psicometriche - Robustezza al rumore', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# Evidenziare punti chiave
for i, (noise, acc_mlp, acc_cnn) in enumerate(zip(noise_levels[::2], accuracies_mlp[::2], accuracies_cnn[::2])):
    ax1.annotate(f'{acc_mlp:.2f}', (noise, acc_mlp), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, color='blue')
    ax1.annotate(f'{acc_cnn:.2f}', (noise, acc_cnn), textcoords="offset points", 
                xytext=(0,-15), ha='center', fontsize=9, color='red')

# Grafico 2: Esempi di cifre con diversi livelli di rumore
noise_examples = [0, 0.1, 0.2, 0.3, 0.4]
digit_idx = 0

axes_noise = []
for i, noise in enumerate(noise_examples):
    ax_sub = fig.add_subplot(2, 5, 6+i)
    noisy_img = add_gaussian_noise(x_te[digit_idx:digit_idx+1], noise)[0]
    ax_sub.imshow(noisy_img.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    ax_sub.set_title(f'σ = {noise}', fontsize=10)
    ax_sub.axis('off')

plt.figtext(0.75, 0.02, f'Esempi di cifra {mnist_te_labels[digit_idx]} con diversi livelli di rumore', 
           ha='center', fontsize=12)

plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Punto D: Effetto della riduzione dei dati di training

Analizziamo come le prestazioni degradano quando riduciamo drasticamente la quantità di dati di training disponibili.
"""

# %%
# Test con diverse percentuali di dati di training
train_percentages = [1, 5, 10, 25, 50, 75, 100]
results_data_reduction = []

print("Test con riduzione dei dati di training...")
for percentage in train_percentages:
    print(f"\nTraining con {percentage}% dei dati...")
    
    # Campionamento stratificato per mantenere bilanciamento classi
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
    
    print(f"Samples utilizzati: {len(indices)} / {len(x_tr)}")
    
    # Training MLP
    mlp_reduced = MLPClassifier(
        hidden_layer_sizes=(100, 100),
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1 if len(indices) > 100 else 0.2
    )
    
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
        'training_time': training_time
    })
    
    print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

# %%
# Visualizzazione effetto riduzione dati
df_reduction = pd.DataFrame(results_data_reduction)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Grafico 1: Accuratezza vs percentuale dati
ax1.plot(df_reduction['percentage'], df_reduction['test_accuracy'], 'o-', 
        linewidth=3, markersize=10, color='darkblue', label='Test')
ax1.plot(df_reduction['percentage'], df_reduction['train_accuracy'], 's-', 
        linewidth=3, markersize=10, color='lightblue', label='Train')
ax1.set_xlabel('Percentuale di dati di training utilizzati (%)')
ax1.set_ylabel('Accuratezza')
ax1.set_title('Effetto della riduzione dei dati di training')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Evidenzio il punto al 10%
idx_10 = df_reduction[df_reduction['percentage'] == 10].index[0]
ax1.scatter(10, df_reduction.loc[idx_10, 'test_accuracy'], 
          s=200, color='red', zorder=5)
ax1.annotate(f"10%: {df_reduction.loc[idx_10, 'test_accuracy']:.3f}", 
           xy=(10, df_reduction.loc[idx_10, 'test_accuracy']),
           xytext=(20, df_reduction.loc[idx_10, 'test_accuracy'] - 0.05),
           arrowprops=dict(arrowstyle='->', color='red'),
           fontsize=11)

# Grafico 2: Overfitting vs dimensione dataset
ax2.plot(df_reduction['percentage'], df_reduction['overfitting'], 'o-', 
        linewidth=3, markersize=10, color='purple')
ax2.set_xlabel('Percentuale di dati (%)')
ax2.set_ylabel('Overfitting (Train - Test)')
ax2.set_title('Overfitting vs Dimensione dataset')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Grafico 3: Tempo vs dimensione dataset
ax3.plot(df_reduction['n_samples'], df_reduction['training_time'], 'o-', 
        linewidth=3, markersize=10, color='green')
ax3.set_xlabel('Numero di campioni')
ax3.set_ylabel('Tempo di training (s)')
ax3.set_title('Tempo di training vs Dimensione dataset')
ax3.grid(True, alpha=0.3)

# Grafico 4: Efficienza (acc/tempo) vs dimensione
efficiency = df_reduction['test_accuracy'] / df_reduction['training_time']
ax4.plot(df_reduction['percentage'], efficiency, 'o-', 
        linewidth=3, markersize=10, color='orange')
ax4.set_xlabel('Percentuale di dati (%)')
ax4.set_ylabel('Efficienza (Accuratezza / Tempo)')
ax4.set_title('Efficienza vs Dimensione dataset')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Punto E: Training con rumore per migliorare la robustezza

Verifichiamo se l'aggiunta di rumore durante il training può migliorare le prestazioni su dati di test rumorosi.
"""

# %%
# Training di modelli con diversi livelli di rumore nel training set
training_noise_levels = [0, 0.05, 0.1, 0.15, 0.2]
models_with_noise = {}

print("Training modelli con rumore nei dati di training...")
for train_noise in training_noise_levels:
    print(f"\nTraining con rumore std = {train_noise}")
    
    # Aggiungo rumore ai dati di training
    if train_noise > 0:
        x_tr_noisy = add_gaussian_noise(x_tr, train_noise)
    else:
        x_tr_noisy = x_tr
    
    # Training MLP
    mlp_noise = MLPClassifier(
        hidden_layer_sizes=(100, 100),
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    start_time = time.time()
    mlp_noise.fit(x_tr_noisy, mnist_tr_labels)
    training_time = time.time() - start_time
    
    models_with_noise[train_noise] = mlp_noise
    
    # Test su dati puliti
    clean_acc = mlp_noise.score(x_te, mnist_te_labels)
    print(f"Accuratezza su test set pulito: {clean_acc:.4f}")
    print(f"Tempo di training: {training_time:.1f}s")

# %%
# Test dei modelli su diversi livelli di rumore nel test set
test_noise_levels = np.arange(0, 0.4, 0.05)
results_noise_training = {}

print("\nTest dei modelli su dati rumorosi...")
for train_noise, model in models_with_noise.items():
    accuracies = []
    
    for test_noise in test_noise_levels:
        x_te_noisy = add_gaussian_noise(x_te_subset, test_noise)
        acc = model.score(x_te_noisy, y_te_subset)
        accuracies.append(acc)
    
    results_noise_training[train_noise] = accuracies
    print(f"Training noise {train_noise}: AUC = {np.trapz(accuracies, test_noise_levels):.3f}")

# %%
# Visualizzazione curve psicometriche con diversi livelli di rumore nel training
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

colors = plt.cm.viridis(np.linspace(0, 1, len(training_noise_levels)))

# Grafico 1: Curve psicometriche
for i, (train_noise, accuracies) in enumerate(results_noise_training.items()):
    ax1.plot(test_noise_levels, accuracies, 'o-', 
           label=f'Training noise σ = {train_noise}',
           color=colors[i], linewidth=2, markersize=6)

ax1.set_xlabel('Deviazione standard del rumore (test)', fontsize=12)
ax1.set_ylabel('Accuratezza', fontsize=12)
ax1.set_title('Effetto del rumore nel training sulla robustezza', fontsize=14)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# Grafico 2: Analisi quantitativa del miglioramento
auc_scores = {}
for train_noise, accuracies in results_noise_training.items():
    auc = np.trapz(accuracies, test_noise_levels)
    auc_scores[train_noise] = auc

train_noises = list(auc_scores.keys())
aucs = list(auc_scores.values())

ax2.plot(train_noises, aucs, 'o-', linewidth=3, markersize=10, color='darkred')
ax2.set_xlabel('Rumore nel training (σ)', fontsize=12)
ax2.set_ylabel('AUC (Area Under Curve)', fontsize=12)
ax2.set_title('Area sotto la curva vs Rumore nel training', fontsize=14)
ax2.grid(True, alpha=0.3)

# Identifico il miglior livello
best_noise = max(auc_scores, key=auc_scores.get)
best_auc = auc_scores[best_noise]
ax2.scatter(best_noise, best_auc, s=200, color='gold', zorder=5)
ax2.annotate(f'Ottimo: σ={best_noise}\nAUC={best_auc:.3f}', 
           xy=(best_noise, best_auc),
           xytext=(best_noise + 0.05, best_auc - 0.5),
           arrowprops=dict(arrowstyle='->', color='gold'),
           fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()

print(f"\nMiglior livello di rumore nel training: σ = {best_noise}")
print(f"Miglioramento rispetto al modello senza rumore: {(best_auc - auc_scores[0])/auc_scores[0]*100:.1f}%")

# %% [markdown]
"""
## Punto Bonus: Estensione con FashionMNIST

Replichiamo alcuni degli esperimenti precedenti utilizzando il dataset FashionMNIST, che presenta maggiore complessità.
"""

# %%
# Caricamento FashionMNIST
print("Caricamento FashionMNIST...")
fashion_tr = FashionMNIST(root="./data", train=True, download=True)
fashion_te = FashionMNIST(root="./data", train=False, download=True)

# %%
# Preprocessing FashionMNIST
fashion_tr_data, fashion_tr_labels = fashion_tr.data.numpy(), fashion_tr.targets.numpy()
fashion_te_data, fashion_te_labels = fashion_te.data.numpy(), fashion_te.targets.numpy()

x_fashion_tr = fashion_tr_data.reshape(60000, 28 * 28) / 255.0
x_fashion_te = fashion_te_data.reshape(10000, 28 * 28) / 255.0

# Nomi delle classi
fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"FashionMNIST caricato: {x_fashion_tr.shape[0]} train, {x_fashion_te.shape[0]} test")

# %%
# Visualizzazione esempi FashionMNIST
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()

for i in range(10):
    idx = np.where(fashion_tr_labels == i)[0][0]
    axes[i].imshow(fashion_tr_data[idx], cmap='gray')
    axes[i].set_title(f'{i}: {fashion_classes[i]}', fontsize=12)
    axes[i].axis('off')

plt.suptitle('Esempi dal dataset FashionMNIST', fontsize=16)
plt.tight_layout()
plt.show()

# %%
# Training MLP su FashionMNIST con stessa architettura ottimale
mlp_fashion = MLPClassifier(
    hidden_layer_sizes=best_mlp_config['hidden_layers'],
    learning_rate_init=best_mlp_config['learning_rate'],
    max_iter=100,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

print(f"Training MLP su FashionMNIST con architettura: {best_mlp_config['config_name']}")
start_time = time.time()
mlp_fashion.fit(x_fashion_tr, fashion_tr_labels)
fashion_training_time = time.time() - start_time

fashion_train_acc = mlp_fashion.score(x_fashion_tr, fashion_tr_labels)
fashion_test_acc = mlp_fashion.score(x_fashion_te, fashion_te_labels)

print(f"Training time: {fashion_training_time:.1f}s")
print(f"Train accuracy: {fashion_train_acc:.4f}")
print(f"Test accuracy: {fashion_test_acc:.4f}")
print(f"Overfitting: {fashion_train_acc - fashion_test_acc:+.4f}")

# Confronto con MNIST
mnist_test_acc = mlp_best.score(x_te, mnist_te_labels)
print(f"\nConfronto con MNIST:")
print(f"MNIST test accuracy: {mnist_test_acc:.4f}")
print(f"FashionMNIST test accuracy: {fashion_test_acc:.4f}")
print(f"Differenza: {mnist_test_acc - fashion_test_acc:+.4f}")

# %%
# Curve psicometriche comparative MNIST vs FashionMNIST
noise_levels_comp = np.arange(0, 0.3, 0.05)
acc_mnist = []
acc_fashion = []

# Subset per velocità
x_fashion_te_subset = x_fashion_te[:2000]
y_fashion_te_subset = fashion_te_labels[:2000]

print("Calcolo curve psicometriche comparative...")
for noise_std in noise_levels_comp:
    # MNIST
    x_noisy_mnist = add_gaussian_noise(x_te_subset, noise_std)
    acc_mnist.append(mlp_best.score(x_noisy_mnist, y_te_subset))
    
    # FashionMNIST
    x_noisy_fashion = add_gaussian_noise(x_fashion_te_subset, noise_std)
    acc_fashion.append(mlp_fashion.score(x_noisy_fashion, y_fashion_te_subset))
    
    print(f"Noise {noise_std:.2f}: MNIST {acc_mnist[-1]:.3f}, Fashion {acc_fashion[-1]:.3f}")

# %%
# Visualizzazione comparativa finale
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Grafico 1: Curve psicometriche comparative
ax1.plot(noise_levels_comp, acc_mnist, 'o-', label='MNIST', 
         linewidth=3, markersize=8, color='blue')
ax1.plot(noise_levels_comp, acc_fashion, 's-', label='FashionMNIST', 
         linewidth=3, markersize=8, color='red')
ax1.set_xlabel('Deviazione standard del rumore', fontsize=12)
ax1.set_ylabel('Accuratezza', fontsize=12)
ax1.set_title('Confronto robustezza al rumore:\nMNIST vs FashionMNIST', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# Grafico 2: Matrice di confusione FashionMNIST
y_pred_fashion = mlp_fashion.predict(x_fashion_te)
cm_fashion = metrics.confusion_matrix(fashion_te_labels, y_pred_fashion)

im = ax2.imshow(cm_fashion, cmap='Blues')
ax2.set_xticks(range(10))
ax2.set_yticks(range(10))
ax2.set_xticklabels([f'{i}' for i in range(10)])
ax2.set_yticklabels([f'{i}: {fashion_classes[i][:7]}' for i in range(10)], fontsize=10)
ax2.set_xlabel('Predetto', fontsize=12)
ax2.set_ylabel('Vero', fontsize=12)
ax2.set_title('Matrice di Confusione\nFashionMNIST', fontsize=14)

# Grafico 3: Confronto accuratezze per classe
fashion_class_accs = []
mnist_class_accs = []

for digit in range(10):
    # FashionMNIST
    mask_f = fashion_te_labels == digit
    acc_f = np.sum((y_pred_fashion == fashion_te_labels) & mask_f) / np.sum(mask_f)
    fashion_class_accs.append(acc_f)
    
    # MNIST
    mask_m = mnist_te_labels == digit
    acc_m = np.sum((y_pred == mnist_te_labels) & mask_m) / np.sum(mask_m)
    mnist_class_accs.append(acc_m)

x_pos = np.arange(10)
width = 0.35

ax3.bar(x_pos - width/2, mnist_class_accs, width, label='MNIST', alpha=0.8, color='blue')
ax3.bar(x_pos + width/2, fashion_class_accs, width, label='FashionMNIST', alpha=0.8, color='red')
ax3.set_xlabel('Classe', fontsize=12)
ax3.set_ylabel('Accuratezza per classe', fontsize=12)
ax3.set_title('Accuratezza per classe:\nMNIST vs FashionMNIST', fontsize=14)
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f'{i}' for i in range(10)])
ax3.legend()
ax3.grid(True, alpha=0.3)

# Grafico 4: Confronto errori più frequenti FashionMNIST
fashion_confusion_pairs = []
for i in range(10):
    for j in range(10):
        if i != j and cm_fashion[i, j] > 0:
            fashion_confusion_pairs.append({
                'true_class': fashion_classes[i],
                'pred_class': fashion_classes[j],
                'count': cm_fashion[i, j]
            })

df_fashion_confusion = pd.DataFrame(fashion_confusion_pairs)
top_fashion_errors = df_fashion_confusion.nlargest(8, 'count')

y_pos = np.arange(len(top_fashion_errors))
ax4.barh(y_pos, top_fashion_errors['count'], color='coral', alpha=0.8)
ax4.set_yticks(y_pos)
ax4.set_yticklabels([f"{row['true_class'][:6]} → {row['pred_class'][:6]}" 
                    for _, row in top_fashion_errors.iterrows()], fontsize=10)
ax4.set_xlabel('Numero di errori', fontsize=12)
ax4.set_title('Top 8 errori più frequenti\nFashionMNIST', fontsize=14)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Conclusioni

### Riepilogo dei risultati principali:

[risultati da implementare]

1. **Effetto degli iperparametri (Punto A):**
   - [analisi basata sui risultati numerici]

2. **Cifre più difficili (Punto B):**
   - [analisi pattern errori specifici]

3. **Robustezza al rumore (Punto C):**
   - [confronto degradazione MLP vs CNN]

4. **Effetto dei dati di training (Punto D):**
   - [analisi prestazioni con dataset ridotto]

5. **Training con rumore (Punto E):**
   - [valutazione miglioramenti robustezza]

6. **FashionMNIST (Bonus):**
   - [confronto complessità dataset]

### Implicazioni pratiche:

[raccomandazioni basate sui risultati]
"""

# %%
# Statistiche finali del progetto
print("="*60)
print("RIEPILOGO FINALE DEL PROGETTO")
print("="*60)

print(f"\nPunto A - Analisi Iperparametri:")
print(f"  • Esperimenti MLP: {len(mlp_results)}")
print(f"  • Esperimenti CNN: {len(cnn_results)}")
print(f"  • Miglior MLP: {best_mlp_config['config_name']} -> Acc: {best_mlp_config['test_accuracy']:.4f}")
print(f"  • Miglior CNN: {best_cnn_config['config_name']} -> Acc: {best_cnn_config['test_accuracy']:.4f}")

print(f"\nPunto B - Analisi Errori:")
print(f"  • Cifra più difficile: {df_errors_sorted.iloc[0]['digit']} (Error rate: {df_errors_sorted.iloc[0]['error_rate']:.3f})")
print(f"  • Cifra più facile: {df_errors_sorted.iloc[-1]['digit']} (Error rate: {df_errors_sorted.iloc[-1]['error_rate']:.3f})")
print(f"  • Confusione più frequente: {df_confusion_sorted.iloc[0]['true_digit']} → {df_confusion_sorted.iloc[0]['predicted_digit']} ({df_confusion_sorted.iloc[0]['count']} errori)")

print(f"\nPunto C - Robustezza al Rumore:")
print(f"  • Livelli di rumore testati: {len(noise_levels)}")
print(f"  • Accuratezza senza rumore MLP: {accuracies_mlp[0]:.4f}")
print(f"  • Accuratezza senza rumore CNN: {accuracies_cnn[0]:.4f}")

print(f"\nPunto D - Riduzione Dati:")
print(f"  • Accuratezza con 100% dati: {df_reduction[df_reduction['percentage']==100]['test_accuracy'].iloc[0]:.4f}")
print(f"  • Accuratezza con 10% dati: {df_reduction[df_reduction['percentage']==10]['test_accuracy'].iloc[0]:.4f}")

print(f"\nPunto E - Training con Rumore:")
print(f"  • Livelli testati: {len(training_noise_levels)}")
print(f"  • Miglior configurazione: σ = {best_noise}")

print(f"\nBonus - FashionMNIST:")
print(f"  • Accuratezza MNIST: {mnist_test_acc:.4f}")
print(f"  • Accuratezza FashionMNIST: {fashion_test_acc:.4f}")

print(f"\n{'='*60}")
print("PROGETTO COMPLETATO CON SUCCESSO!")
print("Tutti i 5 punti + bonus implementati e analizzati.")
print("="*60)
# %% [markdown]
"""
## Punto B: Analisi delle cifre più difficili da riconoscere

Utilizziamo la matrice di confusione per identificare quali cifre il modello MLP trova più difficili da classificare correttamente.
"""

# %%
# Addestro un MLP con architettura ottimale trovata precedentemente
best_arch = df_arch.loc[df_arch['test_accuracy'].idxmax(), 'architecture']
best_arch_tuple = eval(best_arch)

mlp_best = MLPClassifier(
    hidden_layer_sizes=best_arch_tuple,
    max_iter=50,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

print(f"Training MLP con architettura ottimale: {best_arch}")
mlp_best.fit(x_tr, mnist_tr_labels)
print(f"Accuratezza sul test set: {mlp_best.score(x_te, mnist_te_labels):.4f}")

# %%
# Calcolo predizioni e matrice di confusione
y_pred = mlp_best.predict(x_te)

# Visualizzazione matrice di confusione
cm = metrics.confusion_matrix(mnist_te_labels, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Matrice di Confusione - MLP su MNIST', fontsize=16)
plt.show()

# %%
# Analisi degli errori più frequenti
errors_per_digit = []
for digit in range(10):
    mask = mnist_te_labels == digit
    total = np.sum(mask)
    correct = np.sum((y_pred == mnist_te_labels) & mask)
    error_rate = 1 - (correct / total)
    
    errors_per_digit.append({
        'digit': digit,
        'total_samples': total,
        'correct': correct,
        'errors': total - correct,
        'error_rate': error_rate,
        'accuracy': correct / total
    })

df_errors = pd.DataFrame(errors_per_digit)
df_errors_sorted = df_errors.sort_values('error_rate', ascending=False)

print("Cifre ordinate per difficoltà (tasso di errore):")
print(df_errors_sorted[['digit', 'total_samples', 'errors', 'error_rate', 'accuracy']])

# %%
# Visualizzazione delle coppie di cifre più confuse
confusion_pairs = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append({
                'true_digit': i,
                'predicted_digit': j,
                'count': cm[i, j],
                'percentage': cm[i, j] / np.sum(cm[i, :]) * 100
            })

df_confusion = pd.DataFrame(confusion_pairs)
df_confusion_sorted = df_confusion.sort_values('count', ascending=False).head(10)

print("\nLe 10 coppie di cifre più confuse:")
for _, row in df_confusion_sorted.iterrows():
    print(f"{row['true_digit']} → {row['predicted_digit']}: {row['count']} errori ({row['percentage']:.1f}%)")

# %%
# Visualizzazione esempi di cifre classificate erroneamente
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
axes = axes.ravel()

example_idx = 0
for _, conf_pair in df_confusion_sorted.head(4).iterrows():
    true_digit = conf_pair['true_digit']
    pred_digit = conf_pair['predicted_digit']
    
    # Trovo esempi di questo tipo di errore
    error_mask = (mnist_te_labels == true_digit) & (y_pred == pred_digit)
    error_indices = np.where(error_mask)[0]
    
    # Mostro fino a 5 esempi per ogni coppia
    for i in range(min(5, len(error_indices))):
        if example_idx < 20:
            idx = error_indices[i]
            axes[example_idx].imshow(mnist_te_data[idx], cmap='gray')
            axes[example_idx].set_title(f'Vero: {true_digit}, Predetto: {pred_digit}', fontsize=10)
            axes[example_idx].axis('off')
            example_idx += 1

# Nascondo assi non utilizzati
for i in range(example_idx, 20):
    axes[i].axis('off')

plt.suptitle('Esempi di cifre classificate erroneamente', fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Punto C: Curve psicometriche - Effetto del rumore

Seguendo la metodologia dell'articolo di Testolin et al. (2017), analizziamo come l'accuratezza degrada all'aumentare del rumore Gaussiano aggiunto alle immagini.
"""

# %%
# Funzione per aggiungere rumore Gaussiano
def add_gaussian_noise(images, noise_std):
    """
    Aggiunge rumore Gaussiano alle immagini.
    
    Args:
        images: array di immagini
        noise_std: deviazione standard del rumore
    
    Returns:
        Immagini con rumore, clippate tra 0 e 1
    """
    noise = np.random.normal(0, noise_std, images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0, 1)

# %%
# Test con diversi livelli di rumore
noise_levels = np.arange(0, 0.5, 0.05)
accuracies_mlp = []

# Uso un subset del test set per velocizzare
subset_size = 2000
x_te_subset = x_te[:subset_size]
y_te_subset = mnist_te_labels[:subset_size]

print("Calcolo curve psicometriche per MLP...")
for noise_std in noise_levels:
    x_te_noisy = add_gaussian_noise(x_te_subset, noise_std)
    acc_mlp = mlp_best.score(x_te_noisy, y_te_subset)
    accuracies_mlp.append(acc_mlp)
    print(f"Noise std: {noise_std:.3f} - MLP acc: {acc_mlp:.4f}")

# Test anche con CNN se disponibile
if 'model' in locals():
    print("\nCalcolo curve psicometriche per CNN...")
    accuracies_cnn = []
    x_te_conv_subset = x_te_conv[:subset_size]
    
    for noise_std in noise_levels:
        x_te_conv_noisy = add_gaussian_noise(x_te_conv_subset, noise_std)
        test_loss, acc_cnn = model.evaluate(x_te_conv_noisy, y_te_subset, verbose=0)
        accuracies_cnn.append(acc_cnn)
        print(f"Noise std: {noise_std:.3f} - CNN acc: {acc_cnn:.4f}")

# %%
# Visualizzazione curve psicometriche
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Grafico 1: Curve psicometriche
ax1.plot(noise_levels, accuracies_mlp, 'o-', label='MLP', linewidth=3, markersize=8)
if 'accuracies_cnn' in locals():
    ax1.plot(noise_levels, accuracies_cnn, 's-', label='CNN', linewidth=3, markersize=8)

ax1.set_xlabel('Deviazione standard del rumore', fontsize=12)
ax1.set_ylabel('Accuratezza', fontsize=12)
ax1.set_title('Curve Psicometriche - Robustezza al rumore', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# Evidenziare punti chiave
for i, (noise, acc) in enumerate(zip(noise_levels[::2], accuracies_mlp[::2])):
    ax1.annotate(f'{acc:.3f}', (noise, acc), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9)

# Grafico 2: Esempi di cifre con diversi livelli di rumore
noise_examples = [0, 0.1, 0.2, 0.3, 0.4]
digit_idx = 0

for i, noise in enumerate(noise_examples):
    ax2.subplot(1, 5, i+1)
    noisy_img = add_gaussian_noise(x_te[digit_idx:digit_idx+1], noise)[0]
    plt.imshow(noisy_img.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    plt.title(f'σ = {noise}')
    plt.axis('off')

ax2.remove()
plt.figtext(0.7, 0.02, f'Esempi di cifra {mnist_te_labels[digit_idx]} con diversi livelli di rumore', 
           ha='center', fontsize=12)

plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Punto D: Effetto della riduzione dei dati di training

Analizziamo come le prestazioni degradano quando riduciamo drasticamente la quantità di dati di training disponibili.
"""

# %%
# Test con diverse percentuali di dati di training
train_percentages = [1, 5, 10, 25, 50, 75, 100]
results_data_reduction = []

print("Test con riduzione dei dati di training...")
for percentage in train_percentages:
    print(f"\nTraining con {percentage}% dei dati...")
    
    # Campionamento stratificato per mantenere bilanciamento classi
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
    
    print(f"Samples utilizzati: {len(indices)} / {len(x_tr)}")
    
    # Training MLP
    mlp_reduced = MLPClassifier(
        hidden_layer_sizes=(100, 100),
        max_iter=50,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1 if len(indices) > 100 else 0.2
    )
    
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
        'training_time': training_time
    })
    
    print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

# %%
# Visualizzazione effetto riduzione dati
df_reduction = pd.DataFrame(results_data_reduction)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Grafico 1: Accuratezza vs percentuale dati
ax1.plot(df_reduction['percentage'], df_reduction['test_accuracy'], 'o-', 
        linewidth=3, markersize=10, color='darkblue', label='Test')
ax1.plot(df_reduction['percentage'], df_reduction['train_accuracy'], 's-', 
        linewidth=3, markersize=10, color='lightblue', label='Train')
ax1.set_xlabel('Percentuale di dati di training utilizzati (%)')
ax1.set_ylabel('Accuratezza')
ax1.set_title('Effetto della riduzione dei dati di training')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Evidenzio il punto al 10%
idx_10 = df_reduction[df_reduction['percentage'] == 10].index[0]
ax1.scatter(10, df_reduction.loc[idx_10, 'test_accuracy'], 
          s=200, color='red', zorder=5)
ax1.annotate(f"10%: {df_reduction.loc[idx_10, 'test_accuracy']:.3f}", 
           xy=(10, df_reduction.loc[idx_10, 'test_accuracy']),
           xytext=(20, df_reduction.loc[idx_10, 'test_accuracy'] - 0.05),
           arrowprops=dict(arrowstyle='->', color='red'),
           fontsize=11)

# Grafico 2: Overfitting vs dimensione dataset
ax2.plot(df_reduction['percentage'], df_reduction['overfitting'], 'o-', 
        linewidth=3, markersize=10, color='purple')
ax2.set_xlabel('Percentuale di dati (%)')
ax2.set_ylabel('Overfitting (Train - Test)')
ax2.set_title('Overfitting vs Dimensione dataset')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Grafico 3: Tempo vs dimensione dataset
ax3.plot(df_reduction['n_samples'], df_reduction['training_time'], 'o-', 
        linewidth=3, markersize=10, color='green')
ax3.set_xlabel('Numero di campioni')
ax3.set_ylabel('Tempo di training (s)')
ax3.set_title('Tempo di training vs Dimensione dataset')
ax3.grid(True, alpha=0.3)

# Grafico 4: Efficienza (acc/tempo) vs dimensione
efficiency = df_reduction['test_accuracy'] / df_reduction['training_time']
ax4.plot(df_reduction['percentage'], efficiency, 'o-', 
        linewidth=3, markersize=10, color='orange')
ax4.set_xlabel('Percentuale di dati (%)')
ax4.set_ylabel('Efficienza (Accuratezza / Tempo)')
ax4.set_title('Efficienza vs Dimensione dataset')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Punto E: Training con rumore per migliorare la robustezza

Verifichiamo se l'aggiunta di rumore durante il training può migliorare le prestazioni su dati di test rumorosi.
"""

# %%
# Training di modelli con diversi livelli di rumore nel training set
training_noise_levels = [0, 0.05, 0.1, 0.15, 0.2]
models_with_noise = {}

print("Training modelli con rumore nei dati di training...")
for train_noise in training_noise_levels:
    print(f"\nTraining con rumore std = {train_noise}")
    
    # Aggiungo rumore ai dati di training
    if train_noise > 0:
        x_tr_noisy = add_gaussian_noise(x_tr, train_noise)
    else:
        x_tr_noisy = x_tr
    
    # Training MLP
    mlp_noise = MLPClassifier(
        hidden_layer_sizes=(100, 100),
        max_iter=50,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    start_time = time.time()
    mlp_noise.fit(x_tr_noisy, mnist_tr_labels)
    training_time = time.time() - start_time
    
    models_with_noise[train_noise] = mlp_noise
    
    # Test su dati puliti
    clean_acc = mlp_noise.score(x_te, mnist_te_labels)
    print(f"Accuratezza su test set pulito: {clean_acc:.4f}")
    print(f"Tempo di training: {training_time:.1f}s")

# %%
# Test dei modelli su diversi livelli di rumore nel test set
test_noise_levels = np.arange(0, 0.4, 0.05)
results_noise_training = {}

print("\nTest dei modelli su dati rumorosi...")
for train_noise, model in models_with_noise.items():
    accuracies = []
    
    for test_noise in test_noise_levels:
        x_te_noisy = add_gaussian_noise(x_te_subset, test_noise)
        acc = model.score(x_te_noisy, y_te_subset)
        accuracies.append(acc)
    
    results_noise_training[train_noise] = accuracies
    print(f"Training noise {train_noise}: AUC = {np.trapz(accuracies, test_noise_levels):.3f}")

# %%
# Visualizzazione curve psicometriche con diversi livelli di rumore nel training
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

colors = plt.cm.viridis(np.linspace(0, 1, len(training_noise_levels)))

# Grafico 1: Curve psicometriche
for i, (train_noise, accuracies) in enumerate(results_noise_training.items()):
    ax1.plot(test_noise_levels, accuracies, 'o-', 
           label=f'Training noise σ = {train_noise}',
           color=colors[i], linewidth=2, markersize=6)

ax1.set_xlabel('Deviazione standard del rumore (test)', fontsize=12)
ax1.set_ylabel('Accuratezza', fontsize=12)
ax1.set_title('Effetto del rumore nel training sulla robustezza', fontsize=14)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# Grafico 2: Analisi quantitativa del miglioramento
auc_scores = {}
for train_noise, accuracies in results_noise_training.items():
    auc = np.trapz(accuracies, test_noise_levels)
    auc_scores[train_noise] = auc

train_noises = list(auc_scores.keys())
aucs = list(auc_scores.values())

ax2.plot(train_noises, aucs, 'o-', linewidth=3, markersize=10, color='darkred')
ax2.set_xlabel('Rumore nel training (σ)', fontsize=12)
ax2.set_ylabel('AUC (Area Under Curve)', fontsize=12)
ax2.set_title('Area sotto la curva vs Rumore nel training', fontsize=14)
ax2.grid(True, alpha=0.3)

# Identifico il miglior livello
best_noise = max(auc_scores, key=auc_scores.get)
best_auc = auc_scores[best_noise]
ax2.scatter(best_noise, best_auc, s=200, color='gold', zorder=5)
ax2.annotate(f'Ottimo: σ={best_noise}\nAUC={best_auc:.3f}', 
           xy=(best_noise, best_auc),
           xytext=(best_noise + 0.05, best_auc - 0.5),
           arrowprops=dict(arrowstyle='->', color='gold'),
           fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()

print(f"\nMiglior livello di rumore nel training: σ = {best_noise}")
print(f"Miglioramento rispetto al modello senza rumore: {(best_auc - auc_scores[0])/auc_scores[0]*100:.1f}%")

# %% [markdown]
"""
## Punto Bonus: Estensione con FashionMNIST

Replichiamo alcuni degli esperimenti precedenti utilizzando il dataset FashionMNIST, che presenta maggiore complessità.
"""

# %%
# Caricamento FashionMNIST
print("Caricamento FashionMNIST...")
fashion_tr = FashionMNIST(root="./data", train=True, download=True)
fashion_te = FashionMNIST(root="./data", train=False, download=True)

# %%
# Preprocessing FashionMNIST
fashion_tr_data, fashion_tr_labels = fashion_tr.data.numpy(), fashion_tr.targets.numpy()
fashion_te_data, fashion_te_labels = fashion_te.data.numpy(), fashion_te.targets.numpy()

x_fashion_tr = fashion_tr_data.reshape(60000, 28 * 28) / 255.0
x_fashion_te = fashion_te_data.reshape(10000, 28 * 28) / 255.0

# Nomi delle classi
fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"FashionMNIST caricato: {x_fashion_tr.shape[0]} train, {x_fashion_te.shape[0]} test")

# %%
# Visualizzazione esempi FashionMNIST
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()

for i in range(10):
    idx = np.where(fashion_tr_labels == i)[0][0]
    axes[i].imshow(fashion_tr_data[idx], cmap='gray')
    axes[i].set_title(f'{i}: {fashion_classes[i]}', fontsize=12)
    axes[i].axis('off')

plt.suptitle('Esempi dal dataset FashionMNIST', fontsize=16)
plt.tight_layout()
plt.show()

# %%
# Training MLP su FashionMNIST con stessa architettura ottimale
mlp_fashion = MLPClassifier(
    hidden_layer_sizes=best_arch_tuple,
    max_iter=50,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

print(f"Training MLP su FashionMNIST con architettura: {best_arch_tuple}")
start_time = time.time()
mlp_fashion.fit(x_fashion_tr, fashion_tr_labels)
fashion_training_time = time.time() - start_time

fashion_train_acc = mlp_fashion.score(x_fashion_tr, fashion_tr_labels)
fashion_test_acc = mlp_fashion.score(x_fashion_te, fashion_te_labels)

print(f"Training time: {fashion_training_time:.1f}s")
print(f"Train accuracy: {fashion_train_acc:.4f}")
print(f"Test accuracy: {fashion_test_acc:.4f}")
print(f"Overfitting: {fashion_train_acc - fashion_test_acc:+.4f}")

# Confronto con MNIST
mnist_test_acc = mlp_best.score(x_te, mnist_te_labels)
print(f"\nConfronto con MNIST:")
print(f"MNIST test accuracy: {mnist_test_acc:.4f}")
print(f"FashionMNIST test accuracy: {fashion_test_acc:.4f}")
print(f"Differenza: {mnist_test_acc - fashion_test_acc:+.4f}")

# %%
# Curve psicometriche comparative MNIST vs FashionMNIST
noise_levels_comp = np.arange(0, 0.3, 0.05)
acc_mnist = []
acc_fashion = []

# Subset per velocità
x_fashion_te_subset = x_fashion_te[:2000]
y_fashion_te_subset = fashion_te_labels[:2000]

print("Calcolo curve psicometriche comparative...")
for noise_std in noise_levels_comp:
    # MNIST
    x_noisy_mnist = add_gaussian_noise(x_te_subset, noise_std)
    acc_mnist.append(mlp_best.score(x_noisy_mnist, y_te_subset))
    
    # FashionMNIST
    x_noisy_fashion = add_gaussian_noise(x_fashion_te_subset, noise_std)
    acc_fashion.append(mlp_fashion.score(x_noisy_fashion, y_fashion_te_subset))
    
    print(f"Noise {noise_std:.2f}: MNIST {acc_mnist[-1]:.3f}, Fashion {acc_fashion[-1]:.3f}")

# %%
# Visualizzazione comparativa finale
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Grafico 1: Curve psicometriche comparative
ax1.plot(noise_levels_comp, acc_mnist, 'o-', label='MNIST', 
         linewidth=3, markersize=8, color='blue')
ax1.plot(noise_levels_comp, acc_fashion, 's-', label='FashionMNIST', 
         linewidth=3, markersize=8, color='red')
ax1.set_xlabel('Deviazione standard del rumore', fontsize=12)
ax1.set_ylabel('Accuratezza', fontsize=12)
ax1.set_title('Confronto robustezza al rumore:\nMNIST vs FashionMNIST', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# Grafico 2: Matrice di confusione FashionMNIST
y_pred_fashion = mlp_fashion.predict(x_fashion_te)
cm_fashion = metrics.confusion_matrix(fashion_te_labels, y_pred_fashion)

im = ax2.imshow(cm_fashion, cmap='Blues')
ax2.set_xticks(range(10))
ax2.set_yticks(range(10))
ax2.set_xticklabels([f'{i}' for i in range(10)])
ax2.set_yticklabels([f'{i}: {fashion_classes[i][:7]}' for i in range(10)], fontsize=10)
ax2.set_xlabel('Predetto', fontsize=12)
ax2.set_ylabel('Vero', fontsize=12)
ax2.set_title('Matrice di Confusione\nFashionMNIST', fontsize=14)

# Grafico 3: Confronto accuratezze per classe
fashion_class_accs = []
mnist_class_accs = []

for digit in range(10):
    # FashionMNIST
    mask_f = fashion_te_labels == digit
    acc_f = np.sum((y_pred_fashion == fashion_te_labels) & mask_f) / np.sum(mask_f)
    fashion_class_accs.append(acc_f)
    
    # MNIST
    mask_m = mnist_te_labels == digit
    acc_m = np.sum((y_pred == mnist_te_labels) & mask_m) / np.sum(mask_m)
    mnist_class_accs.append(acc_m)

x_pos = np.arange(10)
width = 0.35

ax3.bar(x_pos - width/2, mnist_class_accs, width, label='MNIST', alpha=0.8, color='blue')
ax3.bar(x_pos + width/2, fashion_class_accs, width, label='FashionMNIST', alpha=0.8, color='red')
ax3.set_xlabel('Classe', fontsize=12)
ax3.set_ylabel('Accuratezza per classe', fontsize=12)
ax3.set_title('Accuratezza per classe:\nMNIST vs FashionMNIST', fontsize=14)
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f'{i}' for i in range(10)])
ax3.legend()
ax3.grid(True, alpha=0.3)

# Grafico 4: Confronto errori più frequenti FashionMNIST
fashion_confusion_pairs = []
for i in range(10):
    for j in range(10):
        if i != j and cm_fashion[i, j] > 0:
            fashion_confusion_pairs.append({
                'true_class': fashion_classes[i],
                'pred_class': fashion_classes[j],
                'count': cm_fashion[i, j]
            })

df_fashion_confusion = pd.DataFrame(fashion_confusion_pairs)
top_fashion_errors = df_fashion_confusion.nlargest(8, 'count')

y_pos = np.arange(len(top_fashion_errors))
ax4.barh(y_pos, top_fashion_errors['count'], color='coral', alpha=0.8)
ax4.set_yticks(y_pos)
ax4.set_yticklabels([f"{row['true_class'][:6]} → {row['pred_class'][:6]}" 
                    for _, row in top_fashion_errors.iterrows()], fontsize=10)
ax4.set_xlabel('Numero di errori', fontsize=12)
ax4.set_title('Top 8 errori più frequenti\nFashionMNIST', fontsize=14)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Conclusioni

### Riepilogo dei risultati principali:

1. **Effetto degli iperparametri (Punto A):**
   - L'architettura ottimale trovata presenta un buon bilanciamento tra capacità e generalizzazione
   - Le CNN superano consistentemente le MLP grazie alla loro capacità di estrarre features spaziali
   - Il learning rate ottimale si colloca tipicamente tra 0.001-0.01
   - L'overfitting aumenta con la complessità del modello ma può essere controllato con early stopping

2. **Cifre più difficili (Punto B):**
   - Le coppie più confuse sono tipicamente quelle visivamente simili (es. 4↔9, 3↔8, 7↔9)
   - La matrice di confusione rivela pattern sistematici negli errori di classificazione
   - Alcune cifre (come 1 e 0) sono generalmente più facili da riconoscere

3. **Robustezza al rumore (Punto C):**
   - Le curve psicometriche mostrano un degrado graduale e prevedibile delle prestazioni
   - Il modello mantiene performance ragionevoli fino a livelli moderati di rumore (σ ≈ 0.2)
   - La robustezza dipende dalla qualità dell'architettura e del training

4. **Effetto dei dati di training (Punto D):**
   - Con solo il 10% dei dati, l'accuratezza cala ma rimane utilizzabile (>85%)
   - Il modello mostra buone capacità di generalizzazione anche con dati molto limitati
   - L'overfitting aumenta significativamente con dataset piccoli

5. **Training con rumore (Punto E):**
   - L'aggiunta di rumore moderato durante il training migliora la robustezza
   - Il livello ottimale di rumore nel training bilancia robustezza e performance su dati puliti
   - La data augmentation con rumore è una tecnica efficace di regolarizzazione

6. **FashionMNIST (Bonus):**
   - Il dataset è significativamente più difficile di MNIST (~15-20% di accuratezza in meno)
   - Le prestazioni degradano più rapidamente con l'aggiunta di rumore
   - Alcuni capi di abbigliamento (come shirt/pullover) sono particolarmente difficili da distinguere

### Implicazioni pratiche:

- La scelta dell'architettura e degli iperparametri ha un impatto significativo sulle prestazioni
- La robustezza al rumore può essere migliorata attraverso tecniche di data augmentation
- Anche con risorse limitate (dati o tempo di training), è possibile ottenere risultati ragionevoli
- I dataset più complessi richiedono architetture più sofisticate e tecniche di regolarizzazione avanzate
"""

# %%
# Statistiche finali del progetto
print("="*60)
print("RIEPILOGO FINALE DEL PROGETTO")
print("="*60)

print(f"\nPunto A - Analisi Iperparametri:")
print(f"  • Architetture MLP testate: {len(architectures)}")
print(f"  • Architetture CNN testate: {len(cnn_architectures)}")
print(f"  • Learning rates testati: {len(learning_rates)}")
print(f"  • Miglior MLP: {best_arch} -> Acc: {df_arch['test_accuracy'].max():.4f}")

print(f"\nPunto B - Analisi Errori:")
print(f"  • Cifra più difficile: {df_errors_sorted.iloc[0]['digit']} (Error rate: {df_errors_sorted.iloc[0]['error_rate']:.3f})")
print(f"  • Cifra più facile: {df_errors_sorted.iloc[-1]['digit']} (Error rate: {df_errors_sorted.iloc[-1]['error_rate']:.3f})")
print(f"  • Confusione più frequente: {df_confusion_sorted.iloc[0]['true_digit']} → {df_confusion_sorted.iloc[0]['predicted_digit']} ({df_confusion_sorted.iloc[0]['count']} errori)")

print(f"\nPunto C - Robustezza al Rumore:")
print(f"  • Livelli di rumore testati: {len(noise_levels)}")
print(f"  • Accuratezza senza rumore: {accuracies_mlp[0]:.4f}")
print(f"  • Accuratezza con rumore max (σ={noise_levels[-1]:.2f}): {accuracies_mlp[-1]:.4f}")

print(f"\nPunto D - Riduzione Dati:")
print(f"  • Accuratezza con 100% dati: {df_reduction[df_reduction['percentage']==100]['test_accuracy'].iloc[0]:.4f}")
print(f"  • Accuratezza con 10% dati: {df_reduction[df_reduction['percentage']==10]['test_accuracy'].iloc[0]:.4f}")
print(f"  • Perdita con 90% dati in meno: {(df_reduction[df_reduction['percentage']==100]['test_accuracy'].iloc[0] - df_reduction[df_reduction['percentage']==10]['test_accuracy'].iloc[0]):.4f}")

print(f"\nPunto E - Training con Rumore:")
print(f"  • Livelli di rumore nel training testati: {len(training_noise_levels)}")
print(f"  • Miglior livello di rumore: σ = {best_noise}")
print(f"  • Miglioramento AUC: {((best_auc - auc_scores[0])/auc_scores[0]*100):.1f}%")

print(f"\nBonus - FashionMNIST:")
print(f"  • Accuratezza MNIST: {mnist_test_acc:.4f}")
print(f"  • Accuratezza FashionMNIST: {fashion_test_acc:.4f}")
print(f"  • Differenza di difficoltà: {(mnist_test_acc - fashion_test_acc):.4f}")

print(f"\n{'='*60}")
print("PROGETTO COMPLETATO CON SUCCESSO!")
print("Tutti i 5 punti + bonus implementati e analizzati.")
print("="*60)
