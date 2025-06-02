#!/bin/bash

echo "🚀 Avvio Mini-Progetto IA..."

# Attiva ambiente virtuale
source venv/bin/activate

# Sincronizza file se necessario
echo "🔄 Sincronizzazione file..."
jupytext --sync Mini_progetto_IA.py

# Apri VSCode
echo "💻 Apertura VSCode..."
code .

echo "✅ Ambiente pronto!"
echo ""
echo "📁 File principali:"
echo "   📝 Mini_progetto_IA.py    (per editing)"
echo "   📓 Mini_progetto_IA.ipynb (per esecuzione)"
echo ""
echo "🎯 Workflow:"
echo "   1. Edita il file .py in VSCode"
echo "   2. Salva per aggiornare automaticamente il .ipynb"
echo "   3. Esegui celle nel notebook per vedere i risultati"
