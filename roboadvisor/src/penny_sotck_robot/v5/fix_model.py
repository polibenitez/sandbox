#!/usr/bin/env python3
"""
Script para corregir el flag is_trained del modelo ML
"""
import pickle
import os

model_path = "models/breakout_model.pkl"

print("="*70)
print("CORRIGIENDO MODELO ML")
print("="*70)

if not os.path.exists(model_path):
    print(f"❌ Archivo no existe: {model_path}")
    exit(1)

# Leer modelo actual
with open(model_path, 'rb') as f:
    data = pickle.load(f)

print(f"\n📊 Estado ANTES:")
print(f"   • is_trained: {data.get('is_trained', 'N/A')}")
print(f"   • Modelo: {type(data.get('model'))}")

# Corregir el flag
if data.get('model') is not None:
    data['is_trained'] = True

    # Guardar modelo corregido
    with open(model_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\n✅ Modelo corregido!")
    print(f"   • is_trained: {data['is_trained']}")

    # Verificar
    with open(model_path, 'rb') as f:
        verified = pickle.load(f)

    print(f"\n📊 Estado DESPUÉS:")
    print(f"   • is_trained: {verified.get('is_trained')}")
    print(f"   • Modelo funcional: {verified.get('model') is not None}")

    print(f"\n✅ Corrección completada exitosamente!")
else:
    print(f"\n❌ El modelo es None, no se puede corregir")

print("="*70)
