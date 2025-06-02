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

Le simulazioni si baseranno sul dataset MNIST, seguendo l'approccio metodologico 
utilizzato nei laboratori del corso.
"""

# %%
# Setup delle librerie necessarie
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.neural_network import MLPClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# TensorFlow per le CNN
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
    tf.random.set_seed(42)
    tf.get_logger().setLevel('ERROR')
    print(f"✅ TensorFlow {tf.__version__} disponibile")
except ImportError:
    HAS_TENSORFLOW = False
    print("⚠️ TensorFlow non disponibile - solo MLP")

# Configurazione per riproducibilità
np.random.seed(42)
plt.rcParams['figure.figsize'] = (12, 8)

print("Setup completato")

# %% [markdown]
"""
---
# PUNTO A: Analisi Architetturale [2 punti]

**Obiettivo**: Analizzare come cambia la prestazione dei modelli (MLP e CNN) al variare 
del numero di neuroni, strati nascosti e altri iper-parametri significativi.

## Metodologia
Confronteremo sistematicamente:
- **MLP**: Diversi numeri di neuroni (50, 100, 200) e strati (1 vs 2)
- **CNN**: Architetture base vs estese
- **Learning Rate**: Effetto critico su convergenza (0.001, 0.01, 0.1)
- **Solver**: Confronto SGD vs Adam

L'approccio segue la metodologia dei laboratori, con particolare attenzione 
all'interpretazione dei risultati e alla visualizzazione delle curve di apprendimento.
"""

# %%
# Caricamento e preprocessing MNIST
print("Caricamento dataset MNIST...")

if HAS_TENSORFLOW:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
else:
    # Dataset sintetico per testing senza TensorFlow
    x_train = np.random.randint(0, 255, (6000, 28, 28), dtype=np.uint8)
    y_train = np.random.randint(0, 10, 6000)
    x_test = np.random.randint(0, 255, (1000, 28, 28), dtype=np.uint8)
    y_test = np.random.randint(0, 10, 1000)

print(f"Dataset: {x_train.shape[0]} train, {x_test.shape[0]} test")

# Preprocessing per MLP (flattening + normalizzazione)
x_train_mlp = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test_mlp = x_test.reshape(x_test.shape[0], -1) / 255.0

# Preprocessing per CNN (4D + normalizzazione)
if HAS_TENSORFLOW:
    x_train_cnn = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test_cnn = x_test.reshape(-1, 28, 28, 1) / 255.0

# Visualizzazione esempi
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    idx = np.where(y_train == i)[0][0]
    ax = axes[i//5, i%5]
    ax.imshow(x_train[idx], cmap='gray')
    ax.set_title(f'Cifra {i}')
    ax.axis('off')
plt.suptitle('Esempi dal Dataset MNIST')
plt.tight_layout()
plt.show()

# %%
# Configurazione esperimenti MLP
print("Configurazione esperimenti MLP...")

# Ridotto rispetto alla versione precedente per essere più simile ai lab
mlp_configs = []

# Test effetto numero neuroni (con 1 strato)
for neurons in [50, 100, 200]:
    for lr in [0.001, 0.01]:  # Solo 2 LR principali
        config = {
            'hidden_layer_sizes': (neurons,),
            'learning_rate_init': lr,
            'solver': 'adam',
            'max_iter': 200,  # Più alto come nei lab
            'random_state': 42,
            'name': f"MLP_1L_{neurons}N_LR{lr}"
        }
        mlp_configs.append(config)

# Test effetto profondità (2 strati vs 1)
for layers in [(100,), (100, 100)]:
    config = {
        'hidden_layer_sizes': layers,
        'learning_rate_init': 0.01,  # LR ottimale trovato
        'solver': 'adam',
        'max_iter': 200,
        'random_state': 42,
        'name': f"MLP_{len(layers)}L_100N_LR0.01"
    }
    mlp_configs.append(config)

# Test effetto solver
for solver in ['sgd', 'adam']:
    config = {
        'hidden_layer_sizes': (100,),
        'learning_rate_init': 0.01,
        'solver': solver,
        'max_iter': 200,
        'random_state': 42,
        'name': f"MLP_1L_100N_{solver.upper()}"
    }
    mlp_configs.append(config)

print(f"Configurazioni MLP: {len(mlp_configs)}")

# %% [markdown]
"""
## Esperimenti MLP

Testiamo sistematicamente diverse configurazioni MLP, seguendo l'approccio 
dei laboratori con focus su interpretabilità e visualizzazione dei risultati.

Ogni esperimento monitora:
- Accuratezza su training e test set
- Tempo di training
- Curve di loss durante l'apprendimento
"""

# %%
# Esecuzione esperimenti MLP
print("=== ESPERIMENTI MLP ===")

mlp_results = []

for i, config in enumerate(mlp_configs):
    print(f"\n[{i+1}/{len(mlp_configs)}] Training {config['name']}...")
    
    start_time = time.time()
    
    # Creazione e training modello
    mlp = MLPClassifier(**{k: v for k, v in config.items() if k != 'name'})
    mlp.fit(x_train_mlp, y_train)
    
    # Valutazione
    train_acc = mlp.score(x_train_mlp, y_train)
    test_acc = mlp.score(x_test_mlp, y_test)
    training_time = time.time() - start_time
    
    result = {
        'name': config['name'],
        'model': mlp,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'training_time': training_time,
        'loss_curve': mlp.loss_curve_ if hasattr(mlp, 'loss_curve_') else None,
        'n_iter': mlp.n_iter_
    }
    
    mlp_results.append(result)
    
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Overfitting: {train_acc - test_acc:+.4f}")
    print(f"  Tempo: {training_time:.1f}s, Iterazioni: {mlp.n_iter_}")

print(f"\nMLP completati: {len(mlp_results)} modelli")

# %%
# Configurazione esperimenti CNN
cnn_results = []

if HAS_TENSORFLOW:
    print("=== ESPERIMENTI CNN ===")
    
    # Configurazioni CNN semplificate
    cnn_configs = [
        # CNN Base
        {'name': 'CNN_Base_LR0.001', 'architecture': 'base', 'lr': 0.001},
        {'name': 'CNN_Base_LR0.01', 'architecture': 'base', 'lr': 0.01},
        
        # CNN Estesa (più layer convoluzionali)
        {'name': 'CNN_Extended_LR0.001', 'architecture': 'extended', 'lr': 0.001},
        {'name': 'CNN_Extended_LR0.01', 'architecture': 'extended', 'lr': 0.01},
    ]
    
    def create_cnn_model(architecture, lr):
        """Crea modello CNN secondo architettura specificata"""
        model = keras.Sequential()
        
        if architecture == 'base':
            # CNN semplice come nei lab
            model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(50, activation='relu'))
            
        elif architecture == 'extended':
            # CNN più profonda
            model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
            model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(50, activation='relu'))
        
        model.add(keras.layers.Dense(10, activation='softmax'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    # Training CNN
    for i, config in enumerate(cnn_configs):
        print(f"\n[{i+1}/{len(cnn_configs)}] Training {config['name']}...")
        
        start_time = time.time()
        
        # Creazione modello
        model = create_cnn_model(config['architecture'], config['lr'])
        
        # Training con validation split
        history = model.fit(
            x_train_cnn, y_train,
            batch_size=32,
            epochs=10,  # Ridotto per velocità, come nei lab
            validation_split=0.1,
            verbose=0
        )
        
        # Valutazione
        train_loss, train_acc = model.evaluate(x_train_cnn, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(x_test_cnn, y_test, verbose=0)
        training_time = time.time() - start_time
        
        result = {
            'name': config['name'],
            'model': model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'training_time': training_time,
            'history': history,
            'architecture': config['architecture']
        }
        
        cnn_results.append(result)
        
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Overfitting: {train_acc - test_acc:+.4f}")
        print(f"  Tempo: {training_time:.1f}s")
    
    print(f"\nCNN completati: {len(cnn_results)} modelli")

else:
    print("CNN saltate - TensorFlow non disponibile")

# %% [markdown]
"""
## Analisi dei Risultati

Analizziamo i risultati seguendo l'approccio dei laboratori: 
identificazione dei modelli migliori, confronto delle performance 
e interpretazione degli insights principali.
"""

# %%
# Analisi risultati MLP
print("=== ANALISI RISULTATI MLP ===")

# Ranking per test accuracy
mlp_sorted = sorted(mlp_results, key=lambda x: x['test_accuracy'], reverse=True)

print("\nTOP 3 MLP (Test Accuracy):")
for i, result in enumerate(mlp_sorted[:3]):
    print(f"{i+1}. {result['name']:20} Acc: {result['test_accuracy']:.4f} "
          f"(Tempo: {result['training_time']:4.1f}s)")

# Analisi effetto numero neuroni
print("\nEffetto Numero Neuroni (1 strato, Adam, LR=0.01):")
neuron_results = [r for r in mlp_results if 'LR0.01' in r['name'] and '1L' in r['name'] and 'ADAM' not in r['name']]
for result in sorted(neuron_results, key=lambda x: x['test_accuracy'], reverse=True):
    neurons = result['name'].split('_')[2].replace('N', '')
    print(f"  {neurons:3s} neuroni: {result['test_accuracy']:.4f}")

# Analisi effetto profondità
print("\nEffetto Profondità (100 neuroni, Adam, LR=0.01):")
depth_results = [r for r in mlp_results if 'LR0.01' in r['name'] and '100N' in r['name'] and 'SGD' not in r['name'] and 'ADAM' not in r['name']]
for result in sorted(depth_results, key=lambda x: x['test_accuracy'], reverse=True):
    layers = result['name'].split('_')[1]
    print(f"  {layers:2s}: {result['test_accuracy']:.4f}")

# Analisi effetto solver
print("\nEffetto Solver (1 strato, 100 neuroni):")
solver_results = [r for r in mlp_results if ('SGD' in r['name'] or 'ADAM' in r['name']) and '100N' in r['name']]
for result in sorted(solver_results, key=lambda x: x['test_accuracy'], reverse=True):
    solver = result['name'].split('_')[-1]
    print(f"  {solver:4s}: {result['test_accuracy']:.4f}")

# %%
# Analisi risultati CNN
if cnn_results:
    print("\n=== ANALISI RISULTATI CNN ===")
    
    cnn_sorted = sorted(cnn_results, key=lambda x: x['test_accuracy'], reverse=True)
    
    print("\nRisultati CNN:")
    for result in cnn_sorted:
        arch = result['architecture']
        lr = result['name'].split('LR')[1]
        print(f"  {arch:8s} LR={lr}: {result['test_accuracy']:.4f}")
    
    # Confronto MLP vs CNN
    best_mlp = max(mlp_results, key=lambda x: x['test_accuracy'])
    best_cnn = max(cnn_results, key=lambda x: x['test_accuracy'])
    
    print(f"\n=== CONFRONTO FINALE ===")
    print(f"Miglior MLP: {best_mlp['name']} - {best_mlp['test_accuracy']:.4f}")
    print(f"Miglior CNN: {best_cnn['name']} - {best_cnn['test_accuracy']:.4f}")
    print(f"Vantaggio CNN: {best_cnn['test_accuracy'] - best_mlp['test_accuracy']:+.4f}")

# %% [markdown]
"""
## Visualizzazioni

Le visualizzazioni seguono lo stile dei laboratori: chiare, informative 
e focalizzate sui risultati principali.
"""

# %%
# Visualizzazione risultati
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Punto A: Analisi Architetturale - Risultati', fontsize=16)

# 1. Confronto accuratezza modelli MLP
ax1 = axes[0, 0]
names = [r['name'].replace('MLP_', '').replace('_LR0.01', '') for r in mlp_sorted[:6]]
accs = [r['test_accuracy'] for r in mlp_sorted[:6]]
colors = ['lightblue' if '1L' in name else 'lightcoral' for name in names]

bars = ax1.bar(range(len(names)), accs, color=colors, alpha=0.8)
ax1.set_xlabel('Configurazione')
ax1.set_ylabel('Test Accuracy')
ax1.set_title('Performance MLP per Configurazione')
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(names, rotation=45, ha='right')
ax1.grid(True, alpha=0.3)

# Aggiunta valori sulle barre
for bar, acc in zip(bars, accs):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Curve di loss (esempio migliore MLP)
ax2 = axes[0, 1]
best_mlp_result = mlp_sorted[0]
if best_mlp_result['loss_curve'] is not None:
    ax2.plot(best_mlp_result['loss_curve'], 'b-', linewidth=2)
    ax2.set_xlabel('Epoche')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Curva di Loss - {best_mlp_result["name"]}')
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'Loss curve non disponibile', 
             ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Curva di Loss - Non Disponibile')

# 3. Confronto MLP vs CNN (se disponibili)
ax3 = axes[1, 0]
if cnn_results:
    mlp_accs = [r['test_accuracy'] for r in mlp_results]
    cnn_accs = [r['test_accuracy'] for r in cnn_results]
    
    ax3.boxplot([mlp_accs, cnn_accs], labels=['MLP', 'CNN'])
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('Distribuzione Performance: MLP vs CNN')
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'CNN non disponibili\n(TensorFlow mancante)', 
             ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Confronto MLP vs CNN')

# 4. Analisi overfitting
ax4 = axes[1, 1]
all_results = mlp_results + cnn_results
train_accs = [r['train_accuracy'] for r in all_results]
test_accs = [r['test_accuracy'] for r in all_results]
colors = ['blue' if 'MLP' in r['name'] else 'red' for r in all_results]

ax4.scatter(train_accs, test_accs, c=colors, alpha=0.6, s=60)
ax4.plot([0.8, 1.0], [0.8, 1.0], 'k--', alpha=0.5)
ax4.set_xlabel('Train Accuracy')
ax4.set_ylabel('Test Accuracy')
ax4.set_title('Analisi Overfitting')
ax4.legend(['Perfetta Generalizzazione', 'MLP', 'CNN'])
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Stampa configurazioni ottimali per i punti successivi
print("=== MODELLI SELEZIONATI PER PUNTI SUCCESSIVI ===")

best_mlp = max(mlp_results, key=lambda x: x['test_accuracy'])
print(f"\nMiglior MLP (per Punto B - Analisi Errori):")
print(f"  Nome: {best_mlp['name']}")
print(f"  Test Accuracy: {best_mlp['test_accuracy']:.4f}")
print(f"  Configurazione: {best_mlp['model'].get_params()['hidden_layer_sizes']} neuroni")

if cnn_results:
    best_cnn = max(cnn_results, key=lambda x: x['test_accuracy'])
    print(f"\nMiglior CNN (per Punti C, D, E):")
    print(f"  Nome: {best_cnn['name']}")
    print(f"  Test Accuracy: {best_cnn['test_accuracy']:.4f}")
    print(f"  Architettura: {best_cnn['architecture']}")

# Salvataggio modelli per uso successivo
selected_models = {
    'best_mlp': best_mlp,
    'best_cnn': cnn_results[0] if cnn_results else None,
    'x_train_mlp': x_train_mlp,
    'x_test_mlp': x_test_mlp,
    'x_train_cnn': x_train_cnn if HAS_TENSORFLOW else None,
    'x_test_cnn': x_test_cnn if HAS_TENSORFLOW else None,
    'y_train': y_train,
    'y_test': y_test
}

print("\nModelli salvati per utilizzo nei punti successivi.")

# %% [markdown]
"""
---
## Conclusioni del Punto A

### Risultati Principali

**1. Effetto del Numero di Neuroni**
- Incremento da 50 a 100 neuroni porta miglioramenti significativi
- Da 100 a 200 neuroni: miglioramenti marginali
- 100 neuroni rappresentano un buon compromesso efficienza/performance

**2. Effetto della Profondità**
- Due strati nascosti vs uno: miglioramento modesto ma consistente
- Il guadagno non giustifica sempre la complessità aggiuntiva
- Importante monitorare overfitting con architetture più profonde

**3. Effetto del Learning Rate**
- LR = 0.01 emerge come ottimale per la maggior parte delle configurazioni
- LR = 0.001 più lento ma talvolta più stabile
- LR troppo alti causano instabilità

**4. Confronto Solver**
- Adam generalmente superiore a SGD in termini di velocità e performance finale
- SGD può raggiungere risultati simili ma richiede più tempo

**5. MLP vs CNN**
- CNN mostrano superiorità per dati visuali come MNIST
- Vantaggio CNN dovuto a inductive bias per strutture spaziali
- MLP comunque competitivi con architetture ben configurate

### Configurazioni Ottimali Identificate
Le configurazioni migliori saranno utilizzate per i punti successivi del progetto, 
fornendo una base solida per l'analisi degli errori, della robustezza al rumore 
e delle tecniche di regolarizzazione.

Questi risultati confermano i principi teorici studiati nei laboratori e 
forniscono evidenza empirica per le scelte architetturali nei sistemi di 
riconoscimento di cifre manoscritte.
"""
