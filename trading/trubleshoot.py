# troubleshoot.py
# Script para verificar y solucionar problemas comunes en la aplicación de trading

import os
import sys
import logging
import traceback
import importlib
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"logs/troubleshoot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("troubleshoot")


def check_dependencies():
    """Verifica que todas las dependencias estén instaladas correctamente"""
    logger.info("Verificando dependencias...")

    dependencies = [
        "pandas",
        "numpy",
        "matplotlib",
        "ccxt",
        "talib",
        "scikit-learn",
        "tensorflow",
        "requests",
    ]

    missing = []
    version_info = {}

    for package in dependencies:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "Versión desconocida")
            version_info[package] = version
            logger.info(f"✓ {package} (v{version})")
        except ImportError:
            missing.append(package)
            logger.error(f"✗ {package} - NO INSTALADO")

    if missing:
        logger.error(f"Faltan dependencias: {', '.join(missing)}")
        logger.info("Puedes instalarlas con: pip install " + " ".join(missing))
    else:
        logger.info("Todas las dependencias están instaladas correctamente")

    return missing, version_info


def check_file_structure():
    """Verifica que todos los archivos necesarios estén presentes"""
    logger.info("Verificando estructura de archivos...")

    required_files = [
        "crypto_trading_app.py",
        "advanced_trading_algorithm.py",
        "usage_example.py",
    ]

    missing_files = []

    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
            logger.error(f"✗ {file} - NO ENCONTRADO")
        else:
            logger.info(f"✓ {file}")

    if missing_files:
        logger.error(f"Faltan archivos: {', '.join(missing_files)}")
    else:
        logger.info("Todos los archivos necesarios están presentes")

    # Verificar estructura de carpetas
    if not os.path.exists("logs"):
        logger.warning(
            "Carpeta 'logs' no encontrada. Se creará automáticamente al ejecutar."
        )

    return missing_files


def test_data_download():
    """Intenta descargar datos históricos para verificar conectividad"""
    logger.info("Probando descarga de datos históricos...")

    try:
        from crypto_trading_app import CryptoTradingApp

        app = CryptoTradingApp(mode="test", symbol="BTC/USDT", timeframe="1h")
        app.download_historical_data(days_back=5)  # Solo 5 días para prueba

        if app.data is not None and len(app.data) > 0:
            logger.info(f"✓ Descarga exitosa: {len(app.data)} registros obtenidos")
            logger.info(f"Rango de fechas: {app.data.index[0]} a {app.data.index[-1]}")
            logger.info(f"Último precio: {app.data['close'].iloc[-1]}")
            return True
        else:
            logger.error("✗ La descarga no devolvió datos")
            return False
    except Exception as e:
        logger.error(f"✗ Error al descargar datos: {str(e)}")
        logger.debug(traceback.format_exc())
        return False


def test_indicators():
    """Verifica que los indicadores técnicos se calculen correctamente"""
    logger.info("Probando cálculo de indicadores técnicos...")

    try:
        from crypto_trading_app import CryptoTradingApp

        app = CryptoTradingApp(mode="test", symbol="BTC/USDT", timeframe="1h")
        app.download_historical_data(days_back=5)
        app.apply_indicators()

        # Verificar que todos los indicadores estén presentes
        required_indicators = [
            "sma_20",
            "sma_50",
            "sma_200",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr",
        ]

        missing_indicators = []
        for indicator in required_indicators:
            if indicator not in app.data.columns:
                missing_indicators.append(indicator)
                logger.error(f"✗ Indicador '{indicator}' falta")
            else:
                logger.info(f"✓ Indicador '{indicator}' calculado correctamente")

        if missing_indicators:
            logger.error(f"Faltan indicadores: {', '.join(missing_indicators)}")
            return False
        else:
            logger.info("Todos los indicadores calculados correctamente")

            # Mostrar estadísticas de los indicadores
            for indicator in required_indicators:
                stats = app.data[indicator].describe()
                logger.debug(f"Estadísticas de {indicator}:\n{stats}")

            return True
    except Exception as e:
        logger.error(f"✗ Error al calcular indicadores: {str(e)}")
        logger.debug(traceback.format_exc())
        return False


def test_algorithm():
    """Prueba el algoritmo de trading con un conjunto de datos pequeño"""
    logger.info("Probando algoritmo de trading avanzado...")

    try:
        from crypto_trading_app import CryptoTradingApp
        from advanced_trading_algorithm import AdvancedTradingAlgorithm

        # Descargar datos
        app = CryptoTradingApp(mode="test", symbol="BTC/USDT", timeframe="1h")
        app.download_historical_data(days_back=10)
        app.apply_indicators()

        # Inicializar algoritmo
        algo = AdvancedTradingAlgorithm(base_data=app.data)

        # Probar una decisión de trading
        decision = algo.advanced_trading_algorithm(current_data=app.data)

        logger.info(
            f"Decisión de trading: señal={decision['signal']}, confianza={decision['confidence']:.4f}"
        )
        logger.info(f"Score compuesto: {decision['composite_score']:.4f}")

        # Verificar que la decisión contiene todos los campos necesarios
        required_fields = ["signal", "confidence", "composite_score", "components"]
        missing_fields = []

        for field in required_fields:
            if field not in decision:
                missing_fields.append(field)
                logger.error(f"✗ Campo '{field}' falta en la decisión")
            else:
                logger.info(f"✓ Campo '{field}' presente en la decisión")

        if missing_fields:
            logger.error(f"Faltan campos en la decisión: {', '.join(missing_fields)}")
            return False
        else:
            logger.info("El algoritmo funciona correctamente")
            return True
    except Exception as e:
        logger.error(f"✗ Error al probar el algoritmo: {str(e)}")
        logger.debug(traceback.format_exc())
        return False


def analyze_logs():
    """Analiza los archivos de log existentes para buscar errores comunes"""
    logger.info("Analizando archivos de log existentes...")

    if not os.path.exists("logs"):
        logger.warning("No se encontró la carpeta 'logs'")
        return

    log_files = [f for f in os.listdir("logs") if f.endswith(".log")]

    if not log_files:
        logger.warning("No se encontraron archivos de log para analizar")
        return

    logger.info(f"Analizando {len(log_files)} archivos de log")

    error_patterns = [
        "Error",
        "ERROR",
        "Exception",
        "EXCEPTION",
        "Failed",
        "FAILED",
        "Critical",
        "CRITICAL",
    ]

    error_summary = {}

    for log_file in log_files:
        try:
            with open(os.path.join("logs", log_file), "r") as f:
                content = f.readlines()

                for line in content:
                    for pattern in error_patterns:
                        if pattern in line:
                            if pattern not in error_summary:
                                error_summary[pattern] = []

                            # Extraer el mensaje de error (simplificado)
                            error_msg = (
                                line.split(pattern)[1].strip()
                                if pattern in line.split()
                                else line.strip()
                            )
                            error_summary[pattern].append((log_file, error_msg))
        except Exception as e:
            logger.error(f"Error al analizar archivo de log {log_file}: {str(e)}")

    # Mostrar resumen de errores
    total_errors = sum(len(errors) for errors in error_summary.values())

    if total_errors == 0:
        logger.info("No se encontraron errores en los archivos de log")
        return

    logger.warning(f"Se encontraron {total_errors} errores en los archivos de log")

    # Mostrar los 10 errores más recientes
    recent_errors = []
    for pattern, errors in error_summary.items():
        for log_file, error_msg in errors:
            recent_errors.append((log_file, error_msg))

    recent_errors = sorted(recent_errors, key=lambda x: x[0], reverse=True)[:10]

    logger.info("Errores más recientes:")
    for log_file, error_msg in recent_errors:
        logger.info(f"  [{log_file}] {error_msg}")


def check_internet_connection():
    """Verifica la conexión a Internet y a las APIs necesarias"""
    logger.info("Verificando conexión a Internet...")

    # Lista de endpoints a probar
    endpoints = [
        ("Google", "https://www.google.com"),
        ("Binance API", "https://api.binance.com/api/v3/ping"),
        ("CoinGecko API", "https://api.coingecko.com/api/v3/ping"),
    ]

    all_ok = True

    for name, url in endpoints:
        try:
            logger.info(f"Probando conexión a {name}...")
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"✓ Conexión exitosa a {name}")
            else:
                logger.warning(
                    f"✗ Respuesta inesperada de {name}: código {response.status_code}"
                )
                all_ok = False
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Error al conectar con {name}: {str(e)}")
            all_ok = False

    if all_ok:
        logger.info("Conexión a Internet y APIs verificada correctamente")
    else:
        logger.warning(
            "Se encontraron problemas de conexión. Verifica tu conexión a Internet."
        )

    return all_ok


def fix_common_issues():
    """Intenta solucionar problemas comunes automáticamente"""
    logger.info("Intentando solucionar problemas comunes...")

    # 1. Crear carpeta de logs si no existe
    if not os.path.exists("logs"):
        logger.info("Creando carpeta 'logs'...")
        os.makedirs("logs")
        logger.info("✓ Carpeta 'logs' creada")

    # 2. Verificar permisos de archivos
    logger.info("Verificando permisos de archivos...")
    required_files = [
        "crypto_trading_app.py",
        "advanced_trading_algorithm.py",
        "usage_example.py",
    ]

    for file in required_files:
        if os.path.exists(file):
            try:
                # Intentar abrir en modo lectura y escritura para verificar permisos
                with open(file, "r") as f:
                    pass
                with open(file, "a") as f:
                    pass
                logger.info(f"✓ Permisos correctos para {file}")
            except PermissionError:
                logger.error(f"✗ Error de permisos en {file}")
                try:
                    # Intentar corregir permisos
                    os.chmod(file, 0o644)
                    logger.info(f"✓ Permisos corregidos para {file}")
                except:
                    logger.error(f"No se pudieron corregir los permisos para {file}")

    # 3. Sugerir instalación de dependencias faltantes
    missing, _ = check_dependencies()
    if missing:
        logger.info("Generando comando para instalar dependencias faltantes...")
        cmd = "pip install " + " ".join(missing)
        logger.info(
            f"Ejecuta el siguiente comando para instalar las dependencias faltantes:"
        )
        logger.info(f"  {cmd}")

    logger.info("Proceso de solución de problemas completado")


def main():
    """Función principal para el script de solución de problemas"""
    logger.info("=== INICIANDO DIAGNÓSTICO DE LA APLICACIÓN DE TRADING ===")

    # Ejecutar todas las verificaciones
    check_dependencies()
    print()

    check_file_structure()
    print()

    check_internet_connection()
    print()

    test_data_download()
    print()

    test_indicators()
    print()

    test_algorithm()
    print()

    analyze_logs()
    print()

    # Preguntar si se desea intentar solucionar problemas
    choice = input("\n¿Deseas intentar solucionar problemas automáticamente? (s/n): ")
    if choice.lower() in ["s", "si", "sí", "y", "yes"]:
        fix_common_issues()

    logger.info("=== DIAGNÓSTICO COMPLETADO ===")

    # Resumen final
    logger.info("\nPara ejecutar la aplicación:")
    logger.info("1. Modo normal: python usage_example.py")
    logger.info("2. Modo verboso: python -m logging=DEBUG usage_example.py")
    logger.info("\nRecomendaciones adicionales:")
    logger.info("- Revisa los archivos de log en la carpeta 'logs' para más detalles")
    logger.info(
        "- Si encuentras errores específicos, busca su solución en la documentación"
    )
    logger.info("- Para problemas de API de Binance, verifica tus claves API")


if __name__ == "__main__":
    main()
