# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# INTELLIGENZA ARTIFICIALE
**Prof. Marco Zorzi, Dr. Alberto Testolin**

## Mini-Progetto Individuale - Riconoscimento Cifre Manoscritte

**Nome**: [Inserire nome]  
**Cognome**: [Inserire cognome]  
**Matricola**: [Inserire matricola]  
**Data di consegna**: [Inserire data]

---

### Introduzione e Motivazione

In questo progetto implementeremo un studio sistematico del riconoscimento di cifre manoscritte 
utilizzando il dataset MNIST attraverso diverse architetture di reti neurali. L'obiettivo è 
comprendere come le scelte architetturali e gli iper-parametri influenzano le prestazioni dei 
modelli, seguendo la metodologia consolidata dei laboratori precedenti.

Il progetto si articola in cinque punti principali:
- **Punto A**: Analisi sistematica di architetture MLP e CNN [2 punti]
- **Punto B**: Studio degli errori di classificazione [1 punto]
- **Punto C**: Robustezza al rumore e curve psicometriche [1 punto]
- **Punto D**: Apprendimento con dati limitati [1 punto]
- **Punto E**: Data augmentation con rumore [1 punto]
- **Punto Bonus**: Estensione a FashionMNIST [punto bonus]

L'approccio segue la metodologia dei laboratori precedenti, estendendola con analisi più 
approfondite ispirate alla letteratura scientifica (Testolin et al., 2017).
"""

# %%
# Import delle librerie necessarie
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Machine Learning (seguendo i laboratori precedenti)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    HAS_TF = True
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    HAS_TF = False
    print("TensorFlow non disponibile - usando solo scikit-learn")

# Configurazione per riproducibilità
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
if HAS_TF:
    tf.random.set_seed(RANDOM_STATE)

# Configurazione plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)

print("Setup completato")
print(f"Scikit-learn: Disponibile")
print(f"TensorFlow: {'Disponibile' if HAS_TF else 'Non disponibile'}")

# %% [markdown]
"""
### Caricamento e Preprocessing del Dataset MNIST

Seguendo l'approccio del laboratorio 3, carichiamo il dataset MNIST e applichiamo le 
trasformazioni necessarie per preparare i dati per l'addestramento delle reti neurali.
"""

# %%
# Caricamento dataset MNIST
print("Caricamento dataset MNIST...")

if HAS_TF:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
else:
    # Fallback per testing senza TensorFlow
    print("Creando dataset sintetico per testing...")
    x_train = np.random.randint(0, 255, (6000, 28, 28), dtype=np.uint8)
    y_train = np.random.randint(0, 10, 6000)
    x_test = np.random.randint(0, 255, (1000, 28, 28), dtype=np.uint8)
    y_test = np.random.randint(0, 10, 1000)

print("Informazioni sul dataset:")
print(f"Training set: {x_train.shape} immagini, {y_train.shape} etichette")
print(f"Test set: {x_test.shape} immagini, {y_test.shape} etichette")
print(f"Classi: {np.unique(y_train)}")

# Preprocessing per MLP: normalizzazione e reshape in vettori 1D
x_train_mlp = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255.0
x_test_mlp = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255.0

# Preprocessing per CNN: normalizzazione e aggiunta dimensione canale
if HAS_TF:
    x_train_cnn = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test_cnn = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

print("Preprocessing completato")
print(f"MLP input shape: {x_train_mlp.shape}")
if HAS_TF:
    print(f"CNN input shape: {x_train_cnn.shape}")

# %%
# Visualizzazione di alcuni esempi dal dataset
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(10):
    # Trova primo esempio di ogni cifra
    idx = np.where(y_train == i)[0][0]
    axes[i].imshow(x_train[idx], cmap='gray')
    axes[i].set_title(f'Cifra: {i}')
    axes[i].axis('off')

plt.suptitle('Esempi dal Dataset MNIST', fontsize=16)
plt.tight_layout()
plt.show()

# Distribuzione delle classi
plt.figure(figsize=(10, 6))
unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique, counts, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Cifra')
plt.ylabel('Numero di Esempi')
plt.title('Distribuzione delle Classi nel Training Set')
plt.xticks(unique)
plt.grid(axis='y', alpha=0.3)
for i, count in enumerate(counts):
    plt.text(i, count + 50, str(count), ha='center', va='bottom')
plt.show()

# %% [markdown]
"""
## Punto A [2 punti]: Analisi Architetturale Sistematica

### Obiettivo
Studiare l'effetto del numero di neuroni e strati nascosti sulle prestazioni di entrambi 
i modelli MLP e CNN, analizzando anche l'impatto di diversi iper-parametri.

### Metodologia
Seguendo l'approccio sistematico dei laboratori precedenti, testiamo:

**MLP (Multi-Layer Perceptron):**
- Numero di strati nascosti: 1, 2
- Numero di neuroni per strato: 64, 128, 256 (potenze di 2)
- Learning rate: 0.001, 0.01, 0.1 (includendo 0.1 per mostrare degradazione)

**CNN (Convolutional Neural Network):**
- Architetture: base (1 conv layer) vs estesa (2 conv layers)
- Neuroni nel layer finale: 64, 128
- Learning rate: 0.001, 0.01, 0.1

**Parametri di convergenza:**
- Max iterazioni: 50
- Early stopping attivo
- Tolerance: 0.001

**Totale esperimenti**: 18 MLP + 12 CNN = 30 configurazioni
"""

# %%
# Funzioni helper per la creazione e training dei modelli

def create_and_train_mlp(x_train, y_train, x_test, y_test, hidden_layers, learning_rate, model_name):
    """
    Crea e addestra un modello MLP con early stopping
    """
    print(f"Addestrando {model_name}...")
    
    start_time = time.time()
    
    # Creazione modello MLP con early stopping
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        learning_rate_init=learning_rate,
        max_iter=50,                    # Limite massimo iterazioni
        tol=0.001,                      # Tolerance per early stopping
        early_stopping=True,            # Attiva early stopping
        validation_fraction=0.1,        # 10% per validation
        n_iter_no_change=10,            # Patience per early stopping
        random_state=RANDOM_STATE,
        solver='adam'
    )
    
    # Training
    mlp.fit(x_train, y_train)
    
    training_time = time.time() - start_time
    
    # Valutazione
    train_acc = mlp.score(x_train, y_train)
    test_acc = mlp.score(x_test, y_test)
    
    # Informazioni sulla convergenza
    converged = mlp.n_iter_ < 50
    
    print(f"Completato! Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, " +
          f"Tempo: {training_time:.1f}s, Iterazioni: {mlp.n_iter_}")
    
    return {
        'model': mlp,
        'model_name': model_name,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'training_time': training_time,
        'n_iterations': mlp.n_iter_,
        'converged': converged,
        'loss_curve': getattr(mlp, 'loss_curve_', None)
    }

def create_and_train_cnn(x_train, y_train, x_test, y_test, architecture, neurons_final, learning_rate, model_name):
    """
    Crea e addestra un modello CNN con early stopping
    """
    if not HAS_TF:
        return None
        
    print(f"Addestrando {model_name}...")
    
    start_time = time.time()
    
    # Creazione modello CNN
    model = keras.Sequential()
    
    if architecture == 'base':
        # Architettura base: 1 conv layer (come lab 3)
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    else:
        # Architettura estesa: 2 conv layers
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(neurons_final, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    # Compilazione
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping callback
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=0
    )
    
    # Training
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=50,                      # Limite massimo epoche
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0
    )
    
    training_time = time.time() - start_time
    
    # Valutazione
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"Completato! Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, " +
          f"Tempo: {training_time:.1f}s, Epoche: {len(history.history['loss'])}")
    
    return {
        'model': model,
        'model_name': model_name,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'training_time': training_time,
        'n_epochs': len(history.history['loss']),
        'converged': len(history.history['loss']) < 50,
        'history': history
    }

print("Funzioni helper definite")

# %%
# Configurazione degli esperimenti

# Configurazioni MLP
mlp_configs = []
layers_options = [1, 2]
neurons_options = [64, 128, 256]  # Potenze di 2
learning_rates = [0.001, 0.01, 0.1]  # Include 0.1 per mostrare degradazione

for layers in layers_options:
    for neurons in neurons_options:
        for lr in learning_rates:
            # Configurazione hidden layers
            if layers == 1:
                hidden_layers = (neurons,)
            else:
                hidden_layers = (neurons, neurons)
            
            config = {
                'layers': layers,
                'neurons': neurons,
                'learning_rate': lr,
                'hidden_layers': hidden_layers,
                'name': f'MLP_{layers}L_{neurons}N_LR{lr}'
            }
            mlp_configs.append(config)

# Configurazioni CNN (solo se TensorFlow disponibile)
cnn_configs = []
if HAS_TF:
    arch_types = ['base', 'extended']
    neurons_final_options = [64, 128]
    
    for arch in arch_types:
        for neurons in neurons_final_options:
            for lr in learning_rates:
                config = {
                    'architecture': arch,
                    'neurons_final': neurons,
                    'learning_rate': lr,
                    'name': f'CNN_{arch}_{neurons}N_LR{lr}'
                }
                cnn_configs.append(config)

print(f"Configurazioni definite:")
print(f"MLP: {len(mlp_configs)} configurazioni")
print(f"CNN: {len(cnn_configs)} configurazioni")
print(f"Totale: {len(mlp_configs) + len(cnn_configs)} esperimenti")

# Mostra prime configurazioni come esempio
print("\nPrime 3 configurazioni MLP:")
for i, config in enumerate(mlp_configs[:3]):
    print(f"  {i+1}. {config['name']}")

if cnn_configs:
    print("\nPrime 3 configurazioni CNN:")
    for i, config in enumerate(cnn_configs[:3]):
        print(f"  {i+1}. {config['name']}")

# %% [markdown]
"""
### Esecuzione degli Esperimenti MLP

Procediamo con l'addestramento sistematico di tutti i modelli MLP. Per ogni configurazione, 
raccogliamo metriche dettagliate di performance e tempo di esecuzione.
"""

# %%
# Esecuzione esperimenti MLP
print("=== ESPERIMENTI MLP ===")
print(f"Inizio esperimenti: {datetime.now().strftime('%H:%M:%S')}")

mlp_results = []

for i, config in enumerate(mlp_configs):
    print(f"\n--- Esperimento {i+1}/{len(mlp_configs)}: {config['name']} ---")
    
    result = create_and_train_mlp(
        x_train_mlp, y_train, x_test_mlp, y_test,
        hidden_layers=config['hidden_layers'],
        learning_rate=config['learning_rate'],
        model_name=config['name']
    )
    
    # Aggiungi configurazione ai risultati
    result.update(config)
    result['model_type'] = 'MLP'
    mlp_results.append(result)

print(f"\nMLP completati: {datetime.now().strftime('%H:%M:%S')}")
print(f"Tempo totale MLP: {sum([r['training_time'] for r in mlp_results]):.1f} secondi")

# %% [markdown]
"""
### Esecuzione degli Esperimenti CNN

Proseguiamo con l'addestramento dei modelli CNN, utilizzando early stopping per una 
convergenza efficiente.
"""

# %%
# Esecuzione esperimenti CNN
cnn_results = []

if HAS_TF and cnn_configs:
    print("\n=== ESPERIMENTI CNN ===")
    
    for i, config in enumerate(cnn_configs):
        print(f"\n--- Esperimento {i+1}/{len(cnn_configs)}: {config['name']} ---")
        
        result = create_and_train_cnn(
            x_train_cnn, y_train_cat, x_test_cnn, y_test_cat,
            architecture=config['architecture'],
            neurons_final=config['neurons_final'],
            learning_rate=config['learning_rate'],
            model_name=config['name']
        )
        
        if result is not None:
            result.update(config)
            result['model_type'] = 'CNN'
            cnn_results.append(result)
    
    print(f"\nCNN completati: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Tempo totale CNN: {sum([r['training_time'] for r in cnn_results]):.1f} secondi")
else:
    print("\nCNN saltati - TensorFlow non disponibile")

# Tempo totale esperimenti
all_results = mlp_results + cnn_results
if all_results:
    print(f"\nTempo totale esperimenti: {sum([r['training_time'] for r in all_results]):.1f} secondi")

# %% [markdown]
"""
### Analisi dei Risultati: Tabelle Riassuntive

Organizziamo i risultati ottenuti in tabelle sistematiche per facilitare l'analisi e il 
confronto tra le diverse configurazioni.
"""

# %%
# Creazione tabelle riassuntive
import pandas as pd

if all_results:
    # Creazione DataFrame per analisi sistematica
    results_data = []
    for r in all_results:
        row = {
            'Model_Type': r['model_type'],
            'Name': r['model_name'],
            'Layers': r.get('layers', 'N/A'),
            'Neurons': r.get('neurons', r.get('neurons_final', 'N/A')),
            'Architecture': r.get('architecture', 'N/A'),
            'Learning_Rate': r['learning_rate'],
            'Train_Accuracy': r['train_accuracy'],
            'Test_Accuracy': r['test_accuracy'],
            'Overfitting_Gap': r['train_accuracy'] - r['test_accuracy'],
            'Training_Time': r['training_time'],
            'Iterations': r.get('n_iterations', r.get('n_epochs', 'N/A')),
            'Converged': r['converged']
        }
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    
    # Ordinamento per accuratezza test (decrescente)
    results_df = results_df.sort_values('Test_Accuracy', ascending=False).reset_index(drop=True)
    
    print("=== TOP 10 CONFIGURAZIONI ===")
    top_cols = ['Name', 'Test_Accuracy', 'Train_Accuracy', 'Overfitting_Gap', 'Training_Time', 'Converged']
    print(results_df[top_cols].head(10).to_string(index=False))
    
    print("\n=== BOTTOM 5 CONFIGURAZIONI ===")
    bottom_cols = ['Name', 'Test_Accuracy', 'Train_Accuracy', 'Overfitting_Gap', 'Converged']
    print(results_df[bottom_cols].tail(5).to_string(index=False))
    
    # Analisi convergenza
    print(f"\n=== ANALISI CONVERGENZA ===")
    total_models = len(all_results)
    converged_models = sum([1 for r in all_results if r['converged']])
    print(f"Modelli converged: {converged_models}/{total_models} ({100*converged_models/total_models:.1f}%)")
    
    # Statistiche per tipo di modello
    if mlp_results:
        mlp_converged = sum([1 for r in mlp_results if r['converged']])
        print(f"MLP converged: {mlp_converged}/{len(mlp_results)} ({100*mlp_converged/len(mlp_results):.1f}%)")
    
    if cnn_results:
        cnn_converged = sum([1 for r in cnn_results if r['converged']])
        print(f"CNN converged: {cnn_converged}/{len(cnn_results)} ({100*cnn_converged/len(cnn_results):.1f}%)")

# %%
# Analisi separata per MLP e CNN
if all_results:
    mlp_df = results_df[results_df['Model_Type'] == 'MLP'].copy()
    cnn_df = results_df[results_df['Model_Type'] == 'CNN'].copy()
    
    print("=== ANALISI MLP ===")
    if not mlp_df.empty:
        best_mlp = mlp_df.iloc[0]
        worst_mlp = mlp_df.iloc[-1]
        
        print("Migliore configurazione MLP:")
        print(f"  Modello: {best_mlp['Name']}")
        print(f"  Test Accuracy: {best_mlp['Test_Accuracy']:.4f}")
        print(f"  Overfitting Gap: {best_mlp['Overfitting_Gap']:.4f}")
        print(f"  Iterazioni: {best_mlp['Iterations']}")
        
        print("\nPeggiore configurazione MLP:")
        print(f"  Modello: {worst_mlp['Name']}")
        print(f"  Test Accuracy: {worst_mlp['Test_Accuracy']:.4f}")
        print(f"  Overfitting Gap: {worst_mlp['Overfitting_Gap']:.4f}")
    
    print("\n=== ANALISI CNN ===")
    if not cnn_df.empty:
        best_cnn = cnn_df.iloc[0]
        worst_cnn = cnn_df.iloc[-1]
        
        print("Migliore configurazione CNN:")
        print(f"  Modello: {best_cnn['Name']}")
        print(f"  Test Accuracy: {best_cnn['Test_Accuracy']:.4f}")
        print(f"  Overfitting Gap: {best_cnn['Overfitting_Gap']:.4f}")
        print(f"  Epoche: {best_cnn['Iterations']}")
        
        print("\nPeggiore configurazione CNN:")
        print(f"  Modello: {worst_cnn['Name']}")
        print(f"  Test Accuracy: {worst_cnn['Test_Accuracy']:.4f}")
        print(f"  Overfitting Gap: {worst_cnn['Overfitting_Gap']:.4f}")
    else:
        print("Nessun risultato CNN disponibile")
    
    # Confronto generale
    print("\n=== CONFRONTO MLP vs CNN ===")
    if not mlp_df.empty:
        print(f"Migliore MLP: {mlp_df['Test_Accuracy'].max():.4f}")
        print(f"Tempo medio MLP: {mlp_df['Training_Time'].mean():.1f}s")
        print(f"Iterazioni medie MLP: {mlp_df['Iterations'].mean():.1f}")
    
    if not cnn_df.empty:
        print(f"Migliore CNN: {cnn_df['Test_Accuracy'].max():.4f}")
        print(f"Tempo medio CNN: {cnn_df['Training_Time'].mean():.1f}s")
        print(f"Epoche medie CNN: {cnn_df['Iterations'].mean():.1f}")

# %% [markdown]
"""
### Visualizzazioni: Effetto degli Iper-parametri

Analizziamo graficamente come i diversi iper-parametri influenzano le prestazioni dei modelli, 
seguendo l'approccio visualizzativo dei laboratori precedenti.
"""

# %%
# Visualizzazioni degli effetti degli iper-parametri
if all_results:
    plt.figure(figsize=(15, 12))
    
    # Subplot 1: Effetto Learning Rate per MLP
    plt.subplot(2, 3, 1)
    if not mlp_df.empty:
        for neurons in [64, 128, 256]:
            for layers in [1, 2]:
                subset = mlp_df[(mlp_df['Neurons'] == neurons) & (mlp_df['Layers'] == layers)]
                if not subset.empty:
                    subset_sorted = subset.sort_values('Learning_Rate')
                    plt.plot(subset_sorted['Learning_Rate'], subset_sorted['Test_Accuracy'], 
                            marker='o', label=f'{layers}L-{neurons}N', linewidth=2)
        
        plt.xlabel('Learning Rate')
        plt.ylabel('Test Accuracy')
        plt.title('MLP: Effetto Learning Rate')
        plt.xscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
    
    # Subplot 2: Effetto Learning Rate per CNN
    plt.subplot(2, 3, 2)
    if not cnn_df.empty:
        for arch in ['base', 'extended']:
            for neurons in [64, 128]:
                subset = cnn_df[(cnn_df['Architecture'] == arch) & (cnn_df['Neurons'] == neurons)]
                if not subset.empty:
                    subset_sorted = subset.sort_values('Learning_Rate')
                    plt.plot(subset_sorted['Learning_Rate'], subset_sorted['Test_Accuracy'], 
                            marker='s', label=f'{arch}-{neurons}N', linewidth=2)
        
        plt.xlabel('Learning Rate')
        plt.ylabel('Test Accuracy')
        plt.title('CNN: Effetto Learning Rate')
        plt.xscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'CNN non disponibili', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('CNN: Non disponibili')
    
    # Subplot 3: Effetto numero neuroni MLP (LR ottimale)
    plt.subplot(2, 3, 3)
    if not mlp_df.empty:
        best_lr = 0.001  # Assumiamo che sia ottimale
        for layers in [1, 2]:
            subset = mlp_df[(mlp_df['Learning_Rate'] == best_lr) & (mlp_df['Layers'] == layers)]
            if not subset.empty:
                subset_sorted = subset.sort_values('Neurons')
                plt.plot(subset_sorted['Neurons'], subset_sorted['Test_Accuracy'], 
                        marker='o', label=f'{layers} Layer(s)', linewidth=2, markersize=8)
        
        plt.xlabel('Numero Neuroni')
        plt.ylabel('Test Accuracy')
        plt.title(f'MLP: Effetto Neuroni (LR={best_lr})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Subplot 4: Analisi Overfitting
    plt.subplot(2, 3, 4)
    if not mlp_df.empty:
        plt.scatter(mlp_df['Train_Accuracy'], mlp_df['Test_Accuracy'], 
                   alpha=0.7, label='MLP', s=60, c='blue')
    if not cnn_df.empty:
        plt.scatter(cnn_df['Train_Accuracy'], cnn_df['Test_Accuracy'], 
                   alpha=0.7, label='CNN', s=60, c='red')
    
    # Linea di riferimento (no overfitting)
    plt.plot([0.8, 1.0], [0.8, 1.0], 'k--', alpha=0.5, label='No Overfitting')
    plt.xlabel('Train Accuracy')
    plt.ylabel('Test Accuracy')
    plt.title('Analisi Overfitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Training Time vs Performance
    plt.subplot(2, 3, 5)
    if not mlp_df.empty:
        plt.scatter(mlp_df['Training_Time'], mlp_df['Test_Accuracy'], 
                   alpha=0.7, label='MLP', s=60, c='blue')
    if not cnn_df.empty:
        plt.scatter(cnn_df['Training_Time'], cnn_df['Test_Accuracy'], 
                   alpha=0.7, label='CNN', s=60, c='red')
    
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Test Accuracy')
    plt.title('Efficiency: Time vs Performance')
    if not mlp_df.empty or not cnn_df.empty:
        plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Convergenza vs Performance
    plt.subplot(2, 3, 6)
    if all_results:
        converged_acc = [r['test_accuracy'] for r in all_results if r['converged']]
        not_converged_acc = [r['test_accuracy'] for r in all_results if not r['converged']]
        
        if converged_acc and not_converged_acc:
            plt.boxplot([converged_acc, not_converged_acc], 
                       labels=['Converged', 'Not Converged'])
        elif converged_acc:
            plt.boxplot([converged_acc], labels=['Converged'])
        elif not_converged_acc:
            plt.boxplot([not_converged_acc], labels=['Not Converged'])
        
        plt.ylabel('Test Accuracy')
        plt.title('Convergenza vs Performance')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %%
# Heatmap delle performance
if all_results:
    plt.figure(figsize=(15, 6))
    
    # Heatmap MLP
    if not mlp_df.empty:
        plt.subplot(1, 2, 1)
        
        # Crea pivot table per heatmap
        mlp_pivot = mlp_df.pivot_table(values='Test_Accuracy', 
                                       index=['Layers', 'Neurons'], 
                                       columns='Learning_Rate', 
                                       aggfunc='mean')
        
        # Crea heatmap manualmente usando matplotlib
        im = plt.imshow(mlp_pivot.values, cmap='RdYlGn', aspect='auto')
        plt.colorbar(im, label='Test Accuracy')
        
        # Aggiungi annotazioni
        for i in range(len(mlp_pivot.index)):
            for j in range(len(mlp_pivot.columns)):
                plt.text(j, i, f'{mlp_pivot.values[i,j]:.3f}', 
                        ha='center', va='center', color='black')
        
        plt.xticks(range(len(mlp_pivot.columns)), mlp_pivot.columns)
        plt.yticks(range(len(mlp_pivot.index)), 
                  [f'{idx[0]}L-{idx[1]}N' for idx in mlp_pivot.index])
        plt.xlabel('Learning Rate')
        plt.ylabel('(Layers, Neurons)')
        plt.title('MLP: Heatmap Performance')
    
    # Heatmap CNN
    if not cnn_df.empty:
        plt.subplot(1, 2, 2)
        
        cnn_pivot = cnn_df.pivot_table(values='Test_Accuracy', 
                                       index=['Architecture', 'Neurons'], 
                                       columns='Learning_Rate', 
                                       aggfunc='mean')
        
        im = plt.imshow(cnn_pivot.values, cmap='RdYlGn', aspect='auto')
        plt.colorbar(im, label='Test Accuracy')
        
        # Aggiungi annotazioni
        for i in range(len(cnn_pivot.index)):
            for j in range(len(cnn_pivot.columns)):
                plt.text(j, i, f'{cnn_pivot.values[i,j]:.3f}', 
                        ha='center', va='center', color='black')
        
        plt.xticks(range(len(cnn_pivot.columns)), cnn_pivot.columns)
        plt.yticks(range(len(cnn_pivot.index)), 
                  [f'{idx[0]}-{idx[1]}N' for idx in cnn_pivot.index])
        plt.xlabel('Learning Rate')
        plt.ylabel('(Architecture, Neurons)')
        plt.title('CNN: Heatmap Performance')
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
"""
### Analisi delle Learning Curves

Analizziamo l'andamento dell'apprendimento per le migliori configurazioni, evidenziando 
l'efficacia dell'early stopping nella convergenza dei modelli.
"""

# %%
# Analisi learning curves per le migliori configurazioni
if all_results:
    # Selezione delle migliori configurazioni
    best_mlp = max([r for r in mlp_results], key=lambda x: x['test_accuracy']) if mlp_results else None
    best_cnn = max([r for r in cnn_results], key=lambda x: x['test_accuracy']) if cnn_results else None
    worst_high_lr = [r for r in all_results if r['learning_rate'] == 0.1]
    worst_high_lr = min(worst_high_lr, key=lambda x: x['test_accuracy']) if worst_high_lr else None
    
    # Conta le configurazioni disponibili
    available_configs = []
    if best_mlp: available_configs.append(('Best MLP', best_mlp))
    if best_cnn: available_configs.append(('Best CNN', best_cnn))
    if worst_high_lr: available_configs.append(('Worst High LR', worst_high_lr))
    
    if available_configs:
        n_configs = len(available_configs)
        fig_rows = (n_configs + 1) // 2
        
        plt.figure(figsize=(15, 5 * fig_rows))
        
        for i, (name, result) in enumerate(available_configs):
            plt.subplot(fig_rows, 2, i+1)
            
            if result['model_type'] == 'MLP' and result.get('loss_curve') is not None:
                # Curve MLP
                epochs = range(1, len(result['loss_curve']) + 1)
                plt.plot(epochs, result['loss_curve'], 'b-', linewidth=2, label='Training Loss')
                plt.xlabel('Iterations')
                plt.ylabel('Loss')
                
            elif result['model_type'] == 'CNN' and result.get('history') is not None:
                # Curve CNN
                history = result['history'].history
                epochs = range(1, len(history['loss']) + 1)
                plt.plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss')
                plt.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
            
            plt.title(f'{name}: {result["model_name"]}\n' +
                     f'Final Test Acc: {result["test_accuracy"]:.4f}, ' +
                     f'LR: {result["learning_rate"]}, ' +
                     f'Converged: {result["converged"]}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Learning Curves: Migliori e Peggiori Configurazioni', fontsize=16, y=1.02)
        plt.show()

# %% [markdown]
"""
### Discussione dei Risultati

#### Effetto del Learning Rate

I risultati mostrano chiaramente l'importanza della scelta del learning rate:

1. **Learning Rate Ottimale (0.001-0.01)**: Le configurazioni con learning rate moderati 
   ottengono le prestazioni migliori, permettendo una convergenza stabile con early stopping 
   efficace.

2. **Degradazione con Learning Rate Alto (0.1)**: Come previsto, il learning rate 0.1 causa 
   una significativa degradazione delle prestazioni, spesso impedendo la convergenza entro 
   le 50 iterazioni/epoche.

#### Efficacia dell'Early Stopping

L'implementazione di early stopping con tolerance 0.001 si è dimostrata efficace:
- La maggior parte dei modelli converge prima del limite massimo di 50 iterazioni
- Riduzione significativa dei tempi di training
- Prevenzione dell'overfitting attraverso la validation

#### Effetto dell'Architettura

**MLP vs CNN**: I risultati confermano la superiorità delle CNN per il riconoscimento di immagini:
- Le CNN raggiungono generalmente accuratezze superiori
- Maggiore robustezza rispetto alle variazioni degli iper-parametri
- Convergenza più stabile con early stopping

**Numero di Neuroni**: L'analisi della progressione 64→128→256 neuroni rivela:
- Miglioramento delle prestazioni fino a un punto di saturazione
- Trade-off tra complessità computazionale e guadagno in accuratezza
- Diminishing returns oltre un certo numero di neuroni

#### Analisi dell'Overfitting

La visualizzazione train vs test accuracy evidenzia:
- L'efficacia dell'early stopping nel controllo dell'overfitting
- Configurazioni ben bilanciate grazie alla validation
- L'importanza del bilanciamento tra capacità del modello e generalizzazione
"""

# %%
# Statistiche finali e selezione modelli per punti successivi
print("=== RISULTATI FINALI PUNTO A ===")
print()

if all_results:
    # Migliori modelli per categoria
    best_overall = max(all_results, key=lambda x: x['test_accuracy'])
    best_mlp_result = max([r for r in mlp_results], key=lambda x: x['test_accuracy']) if mlp_results else None
    best_cnn_result = max([r for r in cnn_results], key=lambda x: x['test_accuracy']) if cnn_results else None
    
    print("MIGLIORI MODELLI IDENTIFICATI:")
    print(f"1. Migliore Overall: {best_overall['model_name']} - Test Acc: {best_overall['test_accuracy']:.4f}")
    if best_mlp_result:
        print(f"2. Migliore MLP: {best_mlp_result['model_name']} - Test Acc: {best_mlp_result['test_accuracy']:.4f}")
    if best_cnn_result:
        print(f"3. Migliore CNN: {best_cnn_result['model_name']} - Test Acc: {best_cnn_result['test_accuracy']:.4f}")
    
    print()
    print("ANALISI LEARNING RATE:")
    for lr in [0.001, 0.01, 0.1]:
        lr_results = [r for r in all_results if r['learning_rate'] == lr]
        if lr_results:
            avg_acc = np.mean([r['test_accuracy'] for r in lr_results])
            converged_pct = 100 * np.mean([r['converged'] for r in lr_results])
            print(f"LR {lr}: Acc media = {avg_acc:.4f}, Convergenza = {converged_pct:.1f}%")
    
    print()
    print("ANALISI COMPLESSITA:")
    if mlp_results:
        for neurons in [64, 128, 256]:
            neuron_results = [r for r in mlp_results if r['neurons'] == neurons]
            if neuron_results:
                avg_acc = np.mean([r['test_accuracy'] for r in neuron_results])
                avg_time = np.mean([r['training_time'] for r in neuron_results])
                avg_iter = np.mean([r['n_iterations'] for r in neuron_results])
                print(f"MLP {neurons} neuroni: Acc = {avg_acc:.4f}, Tempo = {avg_time:.1f}s, Iter = {avg_iter:.1f}")
    
    print()
    print("MODELLI SELEZIONATI PER PUNTI SUCCESSIVI:")
    if best_mlp_result:
        print(f"- Punto B (Analisi Errori): {best_mlp_result['model_name']}")
    if best_cnn_result:
        print(f"- Punto C (Curve Psicometriche): {best_cnn_result['model_name']}")
        print(f"- Punto E (Training con Rumore): {best_cnn_result['model_name']}")
    if best_mlp_result and best_cnn_result:
        print(f"- Punto D (Dataset Ridotto): {best_mlp_result['model_name']} e {best_cnn_result['model_name']}")
    
    # Salva configurazioni per i punti successivi
    selected_models = {
        'best_overall': best_overall,
        'best_mlp': best_mlp_result,
        'best_cnn': best_cnn_result,
        'all_results': all_results
    }
    
    print("\n=== PUNTO A COMPLETATO ===")
    print("Configurazioni ottimali identificate e salvate per i punti successivi.")
    print(f"Early stopping efficace: {sum([r['converged'] for r in all_results])}/{len(all_results)} modelli conversi")
else:
    print("Nessun risultato disponibile")

# %% [markdown]
"""
---

## Conclusioni Punto A

L'analisi sistematica di 30 configurazioni con early stopping ha permesso di identificare:

1. **Architetture Ottimali**: Le CNN superano i MLP per il riconoscimento di cifre, 
   confermando la loro idoneità per task visivi

2. **Iper-parametri Cruciali**: Il learning rate è il parametro più critico, con 0.001-0.01 
   che risultano ottimali per la convergenza

3. **Efficacia Early Stopping**: La tolerance 0.001 e limite 50 iterazioni si sono rivelati 
   efficaci nel ridurre i tempi di training e prevenire overfitting

4. **Trade-off Complessità/Performance**: Aumentare il numero di neuroni migliora le prestazioni 
   fino a un punto di saturazione con diminishing returns

5. **Degradazione con LR Alto**: Il learning rate 0.1 causa significativa degradazione delle 
   performance e problemi di convergenza

Le configurazioni ottimali identificate verranno utilizzate nei punti successivi per l'analisi 
degli errori, lo studio della robustezza al rumore e gli esperimenti con dataset ridotti.

---

**Prossimo passo**: Punto B - Analisi degli errori del miglior modello MLP attraverso matrice 
di confusione e visualizzazione dei pattern misclassificati.
"""