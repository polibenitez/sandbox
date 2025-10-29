#!/usr/bin/env python3
"""
Script para verificar el estado del modelo ML
"""
import sys
import os
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

model_path = "models/breakout_model.pkl"

print("="*70)
print("VERIFICACIÓN DEL MODELO ML")
print("="*70)

if os.path.exists(model_path):
    print(f"\n✅ Archivo existe: {model_path}")
    print(f"   Tamaño: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        print(f"\n📦 Contenido del archivo:")
        print(f"   • Claves: {list(data.keys())}")

        if 'model' in data:
            print(f"   • Modelo: {type(data['model'])}")
            print(f"   • Modelo es None: {data['model'] is None}")

        if 'is_trained' in data:
            print(f"   • is_trained: {data['is_trained']}")
        else:
            print(f"   ⚠️  'is_trained' NO está en el archivo")

        if 'feature_names' in data:
            print(f"   • feature_names: {data['feature_names']}")

        # Verificar si el modelo puede hacer predicciones
        if data.get('model') is not None:
            print(f"\n✅ El modelo existe y está cargado")
            print(f"   Tipo: {type(data['model']).__name__}")

            # Intentar hacer una predicción de prueba
            try:
                import numpy as np
                test_features = np.array([[0.07, 19.0, 3.0, 58, 0.004, 0.015, 0.16, 8, 1, 6.0]])
                prediction = data['model'].predict(test_features)
                proba = data['model'].predict_proba(test_features)
                print(f"   ✅ Puede hacer predicciones")
                print(f"   Test predicción: {prediction[0]}, probabilidad: {proba[0][1]:.2%}")
            except Exception as e:
                print(f"   ❌ Error al hacer predicción: {e}")
        else:
            print(f"\n❌ El modelo es None")

    except Exception as e:
        print(f"\n❌ Error al leer el archivo: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\n❌ Archivo NO existe: {model_path}")

print("\n" + "="*70)
