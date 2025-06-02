# Mini-Progetto Intelligenza Artificiale

**Riconoscimento Cifre Manoscritte con Reti Neurali**

## 🚀 Setup Rapido

### 1. Inizializzazione (Una volta sola)
```bash
python setup.py
```

### 2. Attivazione Ambiente
```bash
# Windows
venv\Scripts\activate

# macOS/Linux  
source venv/bin/activate
```

### 3. Conversione e Esecuzione
```bash
# Converte progetto.py → progetto.ipynb
jupytext --sync progetto.py

# Apri notebook
jupyter lab progetto.ipynb
```

## 📋 Workflow Consigliato

1. **Edita** `progetto.py` con il tuo editor preferito
2. **Salva** il file Python
3. **Sincronizza**: `jupytext --sync progetto.py` 
4. **Esegui** celle nel notebook `progetto.ipynb`

## 📊 Struttura Progetto

```
📁 mini-progetto-ia/
├── 📝 progetto.py          # File principale (editable)
├── 📓 progetto.ipynb       # Notebook (auto-generato)
├── 📋 requirements.txt     # Dipendenze
├── 🔧 setup.py            # Setup automatico
├── 📖 README.md           # Questo file
└── 📁 venv/               # Ambiente virtuale
```

## 🎯 Contenuto Progetto

### ✅ Punti Implementati
- **[A]** Analisi architetturale MLP e CNN [2 punti]
- **[B]** Analisi errori con matrice confusione [1 punto] 
- **[C]** Curve psicometriche con rumore [1 punto]
- **[D]** Training con dataset ridotto (10%) [1 punto]
- **[E]** Data augmentation con rumore [1 punto]
- **[Bonus]** Estensione a FashionMNIST [punto bonus]

### 🔧 Configurazioni Tecniche
- **Early stopping**: max_iter=50, tol=0.001
- **Architetture**: MLP (1-2 strati) + CNN (base/estesa)
- **Iper-parametri**: Learning rates 0.001-0.01-0.1
- **Fallback**: Funziona anche senza TensorFlow

## 🛠️ Requisiti

### Minimi
- Python 3.8+
- 4GB RAM
- 2GB spazio disco

### Librerie Principali
- `numpy`, `matplotlib`, `pandas`
- `scikit-learn` (sempre disponibile)
- `tensorflow` (opzionale ma raccomandato)

## 🆘 Risoluzione Problemi

### TensorFlow non si installa?
Il progetto funziona comunque! Userà solo scikit-learn per MLP.

### Jupyter non si apre?
```bash
pip install jupyter jupyterlab
jupyter lab progetto.ipynb
```

### Conversione non funziona?
```bash
pip install jupytext
jupytext --to notebook progetto.py
```

### Errori di memoria?
Riduci dimensioni batch nella sezione CNN o usa dataset ridotto.

## 📈 Risultati Attesi

- **30 configurazioni** testate sistematicamente
- **Curve psicometriche** per analisi robustezza
- **Confronti quantitativi** MLP vs CNN 
- **Visualizzazioni** complete per ogni punto
- **Documentazione** scientifica rigorosa

## 🏆 Consegna

Il file finale sarà `progetto.ipynb` con:
- Codice funzionante e risultati
- Discussioni teoriche
- Visualizzazioni informative  
- Conclusioni ben argomentate

---

**Tempo stimato**: 15-20 ore totali  
**Difficoltà**: Intermedia  
**Voto atteso**: 6/6 punti + bonus se completato correttamente