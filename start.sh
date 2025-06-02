#!/bin/bash

echo "🚀 Mini-Progetto Intelligenza Artificiale"
echo "=========================================="

# Controlla se l'ambiente virtuale esiste
if [ ! -d "venv" ]; then
    echo "⚠️ Ambiente virtuale non trovato. Eseguire prima:"
    echo "   python setup.py"
    exit 1
fi

# Attiva ambiente virtuale
echo "🔧 Attivazione ambiente virtuale..."
source venv/bin/activate

# Sincronizza file se esiste progetto.py
if [ -f "progetto.py" ]; then
    echo "🔄 Sincronizzazione progetto.py → progetto.ipynb..."
    jupytext --sync progetto.py
    echo "✅ Sincronizzazione completata"
else
    echo "⚠️ progetto.py non trovato"
fi

# Verifica che il notebook esista
if [ -f "progetto.ipynb" ]; then
    echo "📓 Aprendo Jupyter Lab..."
    echo ""
    echo "💡 Una volta aperto Jupyter:"
    echo "   1. Apri progetto.ipynb"
    echo "   2. Esegui le celle"
    echo "   3. Per modifiche, edita progetto.py e riesegui questo script"
    echo ""
    
    # Apri Jupyter Lab
    jupyter lab progetto.ipynb
else
    echo "❌ progetto.ipynb non trovato"
    echo "   Crea prima progetto.py o esegui: jupytext --to notebook progetto.py"
fi