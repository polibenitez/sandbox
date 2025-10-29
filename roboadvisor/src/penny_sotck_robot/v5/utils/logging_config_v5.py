#!/usr/bin/env python3
"""
LOGGING CONFIGURATION V5
========================
Sistema de logging configurable con niveles dinámicos

Features:
- Configuración via logging.config.dictConfig()
- Niveles dinámicos (INFO/DEBUG)
- Múltiples handlers (consola, archivo, rotating)
- Formateo personalizado
"""

import logging
import logging.config
from datetime import datetime
import os


def setup_logging(level: str = "INFO", log_to_file: bool = True,
                  log_dir: str = "./logs") -> None:
    """
    Configura el sistema de logging para V5

    Args:
        level: Nivel de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_to_file: Si True, guarda logs en archivo
        log_dir: Directorio para archivos de log
    """

    # Crear directorio de logs si no existe
    if log_to_file and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Nombre del archivo de log con timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f"trading_v5_{timestamp}.log")

    # Configuración de logging
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            },
            'console': {
                'format': '%(asctime)s | %(levelname)-8s | %(message)s',
                'datefmt': '%H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'console',
                'stream': 'ext://sys.stdout'
            }
        },
        'root': {
            'level': level,
            'handlers': ['console']
        },
        'loggers': {
            'penny_stock_v5': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            },
            'market_data': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            },
            'backtester': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            },
            'ml_model': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            }
        }
    }

    # Agregar handler de archivo si está habilitado
    if log_to_file:
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': level,
            'formatter': 'detailed',
            'filename': log_filename,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }

        # Agregar file handler a todos los loggers
        config['root']['handlers'].append('file')
        for logger_name in config['loggers']:
            config['loggers'][logger_name]['handlers'].append('file')

    # Aplicar configuración
    logging.config.dictConfig(config)

    logger = logging.getLogger('penny_stock_v5')
    logger.info(f"Sistema de logging V5 inicializado - Nivel: {level}")
    if log_to_file:
        logger.info(f"Logs guardándose en: {log_filename}")


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger configurado

    Args:
        name: Nombre del logger

    Returns:
        Logger configurado
    """
    return logging.getLogger(name)


def set_log_level(level: str):
    """
    Cambia el nivel de logging dinámicamente

    Args:
        level: Nuevo nivel ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    logging.getLogger().setLevel(numeric_level)

    logger = logging.getLogger('penny_stock_v5')
    logger.info(f"Nivel de logging cambiado a: {level}")


class LogContext:
    """Context manager para logging temporal con nivel diferente"""

    def __init__(self, logger_name: str, level: str):
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.new_level = getattr(logging, level.upper())

    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


# Decorador para logging de funciones
def log_execution(logger_name: str = 'penny_stock_v5'):
    """
    Decorador que loggea la ejecución de una función

    Usage:
        @log_execution('my_logger')
        def my_function():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name)
            logger.debug(f"Ejecutando {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completado exitosamente")
                return result
            except Exception as e:
                logger.error(f"Error en {func.__name__}: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test del sistema de logging
    setup_logging(level="DEBUG", log_to_file=True)

    logger = get_logger('penny_stock_v5')
    logger.debug("Mensaje de DEBUG")
    logger.info("Mensaje de INFO")
    logger.warning("Mensaje de WARNING")
    logger.error("Mensaje de ERROR")

    # Test context manager
    with LogContext('penny_stock_v5', 'WARNING'):
        logger.debug("Este mensaje NO se mostrará")
        logger.warning("Este mensaje SÍ se mostrará")

    # Test decorador
    @log_execution()
    def test_function():
        logger.info("Dentro de test_function")
        return "resultado"

    test_function()

    print("\nTest completado. Revisa el archivo de log generado en ./logs/")
