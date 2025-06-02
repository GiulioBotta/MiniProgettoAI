#!/usr/bin/env python3
"""
Setup automatico per Mini-Progetto IA
Crea ambiente virtuale e installa dipendenze
"""

import os
import sys
import subprocess
import venv

def run_command(cmd, description):
    """Esegue comando con output"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completato")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore durante {description}: {e.stderr}")
        return False

def main():
    print("🚀 Setup Mini-Progetto Intelligenza Artificiale")
    print("=" * 50)
    
    # Verifica Python
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ richiesto")
        sys.exit(1)
    
    print(f"✅ Python {python_version.major}.{python_version.minor} OK")
    
    # Crea ambiente virtuale
    venv_path = "venv"
    if not os.path.exists(venv_path):
        print(f"🔧 Creando ambiente virtuale in '{venv_path}'...")
        venv.create(venv_path, with_pip=True)
        print("✅ Ambiente virtuale creato")
    else:
        print("✅ Ambiente virtuale già presente")
    
    # Determina comando pip
    if os.name == 'nt':  # Windows
        pip_cmd = f"{venv_path}\\Scripts\\pip"
        python_cmd = f"{venv_path}\\Scripts\\python"
    else:  # Unix/Linux/macOS
        pip_cmd = f"{venv_path}/bin/pip"
        python_cmd = f"{venv_path}/bin/python"
    
    # Aggiorna pip
    run_command(f"{pip_cmd} install --upgrade pip", "Aggiornamento pip")
    
    # Installa dipendenze
    if os.path.exists("requirements.txt"):
        run_command(f"{pip_cmd} install -r requirements.txt", "Installazione dipendenze")
    else:
        print("⚠️ requirements.txt non trovato, installazione manuale...")
        packages = [
            "numpy>=1.21.0",
            "matplotlib>=3.5.0", 
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "tensorflow>=2.8.0",
            "jupyter>=1.0.0",
            "jupytext>=1.13.0",
            "tqdm>=4.62.0"
        ]
        
        for package in packages:
            run_command(f"{pip_cmd} install {package}", f"Installazione {package}")
    
    # Configura jupytext per conversione automatica
    run_command(f"{pip_cmd} install jupytext", "Configurazione jupytext")
    
    # Test installazione
    print("\n🧪 Test installazione...")
    test_cmd = f"{python_cmd} -c \"import numpy, matplotlib, pandas, sklearn; print('✅ Librerie base OK')\""
    run_command(test_cmd, "Test librerie base")
    
    # Test TensorFlow (opzionale)
    tf_test = f"{python_cmd} -c \"import tensorflow as tf; print(f'✅ TensorFlow {{tf.__version__}} OK')\""
    if not run_command(tf_test, "Test TensorFlow"):
        print("⚠️ TensorFlow non disponibile - il progetto funzionerà comunque con scikit-learn")
    
    print(f"\n🎯 SETUP COMPLETATO!")
    print("=" * 50)
    print("📁 File principali:")
    print("   📝 progetto.py      (editing)")
    print("   📓 progetto.ipynb   (esecuzione)")
    print("")
    print("🚀 Per iniziare:")
    if os.name == 'nt':
        print("   1. .\\venv\\Scripts\\activate")
    else:
        print("   1. source venv/bin/activate")
    print("   2. jupytext --sync progetto.py")
    print("   3. jupyter lab progetto.ipynb")
    print("")
    print("💡 Workflow:")
    print("   • Edita progetto.py")
    print("   • Salva per aggiornare progetto.ipynb")
    print("   • Esegui celle nel notebook")

if __name__ == "__main__":
    main()