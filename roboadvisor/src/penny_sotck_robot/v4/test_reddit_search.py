#!/usr/bin/env python3
"""
TEST RÁPIDO - Reddit Search con snscrape
=========================================

Script para probar que snscrape funciona correctamente
con Reddit antes de ejecutar el screener completo.

Uso:
    python test_reddit_search.py
"""

import subprocess
import json
import sys

def test_snscrape_installation():
    """Verifica que snscrape está instalado"""
    print("🔍 Verificando instalación de snscrape...")
    try:
        result = subprocess.run(
            ["snscrape", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ snscrape está instalado")
            print(f"   Versión: {result.stdout.strip()}")
            return True
        else:
            print("❌ snscrape no responde correctamente")
            return False
    except FileNotFoundError:
        print("❌ snscrape NO está instalado")
        print("\n💡 Para instalar:")
        print("   pip install snscrape")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_reddit_search(symbol="CLOV"):
    """Prueba búsqueda de Reddit"""
    print(f"\n🔍 Probando búsqueda de Reddit para: {symbol}")
    print("   (Esto puede tardar 10-30 segundos...)\n")

    try:
        cmd = [
            "snscrape",
            "--jsonl",
            "--max-results", "10",
            "reddit-search",
            symbol
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"⚠️  Comando retornó código: {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr}")
            return False

        lines = result.stdout.strip().split('\n')
        valid_posts = 0

        print(f"📊 Resultados encontrados:\n")

        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue

            try:
                post = json.loads(line)

                valid_posts += 1

                # Mostrar información del post
                title = post.get('title', 'Sin título')[:60]
                url = post.get('url', 'Sin URL')
                subreddit = post.get('subreddit', 'unknown')

                print(f"{i}. 📝 {title}...")
                print(f"   🔗 r/{subreddit}")
                print(f"   🌐 {url}\n")

            except json.JSONDecodeError:
                continue

        if valid_posts == 0:
            print(f"⚠️  No se encontraron posts para '{symbol}'")
            print("   Esto puede ser normal si el símbolo no tiene actividad reciente")
        else:
            print(f"✅ Encontrados {valid_posts} posts válidos")

        return True

    except subprocess.TimeoutExpired:
        print("❌ Timeout - La búsqueda tardó más de 30 segundos")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Función principal"""
    print("="*70)
    print("TEST DE REDDIT SEARCH CON SNSCRAPE")
    print("="*70)

    # Test 1: Verificar instalación
    if not test_snscrape_installation():
        print("\n⛔ No se puede continuar sin snscrape instalado")
        sys.exit(1)

    # Test 2: Buscar símbolo popular
    symbols_to_test = ["CLOV", "GME", "AMC"]

    print(f"\n🧪 Probando con símbolos: {', '.join(symbols_to_test)}")

    for symbol in symbols_to_test:
        success = test_reddit_search(symbol)
        if success:
            break  # Si uno funciona, está bien

        print(f"\n⏭️  Intentando con siguiente símbolo...\n")

    print("\n" + "="*70)
    print("✅ TEST COMPLETADO")
    print("="*70)
    print("\n💡 Si viste posts de Reddit, el sistema funciona correctamente")
    print("   Ahora puedes ejecutar: python search_pennys.py\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrumpido por usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
