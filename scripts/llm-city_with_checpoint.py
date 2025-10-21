import pandas as pd
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import random
import json
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import pickle
import sqlite3

# ==== CONFIGURACIÓN GLOBAL ====
# Configuración de API
API_KEY = "tuapikeyaq"  # Clave API de OpenAI
BASE_URL = "http://localhost:11434/v1"  # URL base de la API
MODEL = "gemma3:4b"  # Modelo a utilizar

# Configuración de procesamiento
FOLDER = "data/jp/"  # Carpeta donde está el archivo y donde se guardarán los resultados
FILE_NAME = "regex.parquet"  # Nombre del archivo a procesar
CITY_COLUMN = "person_address"  # Nombre de la columna que contiene las direcciones
BATCH_SIZE = 100  # Tamaño de lote para procesamiento
MAX_WORKERS = 4  # Número máximo de hilos concurrentes
MAX_RETRIES = 3  # Número máximo de reintentos

# Configuración de archivos
CHECKPOINT_FORMAT = (
    "pickle"  # Formato para guardar el checkpoint: "sqlite", "pickle" o "json"
)
CHECKPOINT_FILE = (
    "city_checkpoint"  # Nombre base del archivo de checkpoint (sin extensión)
)

# Modo de ejecución
RESUME = True  # Si es True, reanuda desde el último checkpoint

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CityValidator:
    """
    Validador de ciudades usando la librería oficial de OpenAI con procesamiento
    concurrente para mayor eficiencia.
    """

    def __init__(
        self,
        api_key,
        base_url=None,
        model="gpt-3.5-turbo",
        max_workers=5,
        max_retries=3,
        batch_size=100,
        checkpoint_file=None,
    ):
        """
        Inicializa el validador de ciudades.

        Args:
            api_key (str): Clave API de OpenAI
            base_url (str, optional): URL base para la API. Por defecto utiliza la de OpenAI.
            model (str): Modelo a utilizar. Por defecto "gpt-3.5-turbo".
            max_workers (int): Número máximo de hilos concurrentes.
            max_retries (int): Número máximo de reintentos para llamadas a la API.
            batch_size (int): Tamaño de lote para procesamiento por lotes.
            checkpoint_file (str): Nombre del archivo de checkpoint para guardar el progreso.
        """
        self.model = model
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.batch_size = batch_size

        # Establecer ruta base del checkpoint
        if checkpoint_file:
            self.checkpoint_base = os.path.join(FOLDER, checkpoint_file)
        else:
            self.checkpoint_base = os.path.join(FOLDER, CHECKPOINT_FILE)

        # Rutas específicas según formato
        if CHECKPOINT_FORMAT == "sqlite":
            self.checkpoint_path = f"{self.checkpoint_base}.db"
        elif CHECKPOINT_FORMAT == "pickle":
            self.checkpoint_path = f"{self.checkpoint_base}.pkl"
        else:  # default to json
            self.checkpoint_path = f"{self.checkpoint_base}.json"

        # Inicializar cliente de OpenAI
        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url

        self.client = OpenAI(**client_args)

        # Cargar el checkpoint si existe
        self.address_to_city_cache = self._load_checkpoint()

        # Si usamos SQLite, también mantenemos un diccionario en memoria para acceso rápido
        if CHECKPOINT_FORMAT == "sqlite":
            self.in_memory_cache = {}
            self.in_memory_cache.update(self.address_to_city_cache)

    def _load_checkpoint(self):
        """
        Carga el checkpoint si existe según el formato configurado.

        Returns:
            dict: Diccionario con los datos cacheados (address -> city)
        """
        # Si no estamos en modo RESUME, no cargar checkpoint
        if not RESUME:
            logger.info(f"Modo nuevo: ignorando checkpoint existente (si hay)")
            return {}

        try:
            if CHECKPOINT_FORMAT == "sqlite":
                return self._load_checkpoint_sqlite()
            elif CHECKPOINT_FORMAT == "pickle":
                return self._load_checkpoint_pickle()
            else:  # default to json
                return self._load_checkpoint_json()
        except Exception as e:
            logger.warning(
                f"Error al cargar checkpoint: {str(e)}. Continuando sin datos cacheados."
            )
            return {}

    def _load_checkpoint_sqlite(self):
        """Carga el checkpoint desde una base de datos SQLite."""
        if not os.path.exists(self.checkpoint_path):
            logger.info(f"No se encontró base de datos SQLite: {self.checkpoint_path}")
            return {}

        try:
            # Conectar a la base de datos
            conn = sqlite3.connect(self.checkpoint_path)
            cursor = conn.cursor()

            # Verificar si existe la tabla
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='cities'"
            )
            if not cursor.fetchone():
                logger.info(
                    "La base de datos existe pero no contiene la tabla 'cities'"
                )
                conn.close()
                return {}

            # Obtener todos los registros
            cursor.execute("SELECT address, city FROM cities")
            results = dict(cursor.fetchall())
            conn.close()

            count = len(results)
            logger.info(
                f"Cargadas {count} direcciones desde SQLite: {self.checkpoint_path}"
            )
            return results
        except Exception as e:
            logger.error(f"Error cargando datos desde SQLite: {str(e)}")
            return {}

    def _load_checkpoint_pickle(self):
        """Carga el checkpoint desde un archivo pickle."""
        if not os.path.exists(self.checkpoint_path):
            logger.info(f"No se encontró archivo pickle: {self.checkpoint_path}")
            return {}

        try:
            with open(self.checkpoint_path, "rb") as f:
                data = pickle.load(f)
                count = len(data)
                logger.info(
                    f"Cargadas {count} direcciones desde pickle: {self.checkpoint_path}"
                )
                return data
        except Exception as e:
            logger.error(f"Error cargando datos desde pickle: {str(e)}")
            return {}

    def _load_checkpoint_json(self):
        """Carga el checkpoint desde un archivo JSON."""
        if not os.path.exists(self.checkpoint_path):
            logger.info(f"No se encontró archivo JSON: {self.checkpoint_path}")
            return {}

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                count = len(data)
                logger.info(
                    f"Cargadas {count} direcciones desde JSON: {self.checkpoint_path}"
                )
                return data
        except Exception as e:
            logger.error(f"Error cargando datos desde JSON: {str(e)}")
            return {}

    def _save_checkpoint(self, data):
        """
        Guarda el checkpoint con los datos procesados según el formato configurado.

        Args:
            data (dict): Diccionario con los datos a guardar (address -> city)
        """
        try:
            # Asegurarse de que la carpeta existe
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

            if CHECKPOINT_FORMAT == "sqlite":
                self._save_checkpoint_sqlite(data)
            elif CHECKPOINT_FORMAT == "pickle":
                self._save_checkpoint_pickle(data)
            else:  # default to json
                self._save_checkpoint_json(data)

        except Exception as e:
            logger.error(f"Error al guardar checkpoint: {str(e)}")

    def _save_checkpoint_sqlite(self, data):
        """Guarda el checkpoint en una base de datos SQLite."""
        try:
            # Si estamos usando un diccionario en memoria, actualizar con los nuevos datos
            if hasattr(self, "in_memory_cache"):
                new_items = {
                    k: v for k, v in data.items() if k not in self.in_memory_cache
                }
                if not new_items:
                    # No hay nuevos datos para guardar
                    return
                self.in_memory_cache.update(new_items)
                data_to_save = new_items
            else:
                data_to_save = data

            # Conectar a la base de datos
            conn = sqlite3.connect(self.checkpoint_path)
            cursor = conn.cursor()

            # Crear tabla si no existe
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS cities (
                address TEXT PRIMARY KEY,
                city TEXT NOT NULL
            )
            """
            )

            # Insertar o actualizar datos
            for address, city in data_to_save.items():
                cursor.execute(
                    "INSERT OR REPLACE INTO cities (address, city) VALUES (?, ?)",
                    (address, city),
                )

            # Confirmar cambios y cerrar
            conn.commit()
            conn.close()

            logger.info(
                f"Checkpoint SQLite guardado con {len(data_to_save)} nuevos elementos"
            )
        except Exception as e:
            logger.error(f"Error guardando datos en SQLite: {str(e)}")

    def _save_checkpoint_pickle(self, data):
        """Guarda el checkpoint en un archivo pickle."""
        try:
            with open(self.checkpoint_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Checkpoint pickle guardado con {len(data)} elementos")
        except Exception as e:
            logger.error(f"Error guardando datos en pickle: {str(e)}")

    def _save_checkpoint_json(self, data):
        """Guarda el checkpoint en un archivo JSON."""
        try:
            with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            logger.info(f"Checkpoint JSON guardado con {len(data)} elementos")
        except Exception as e:
            logger.error(f"Error guardando datos en JSON: {str(e)}")

    def _get_city_from_cache(self, address):
        """
        Obtiene una ciudad del cache según el formato configurado.

        Args:
            address (str): Dirección a buscar

        Returns:
            str: Ciudad si existe en cache, None si no existe
        """
        address_str = str(address)

        if CHECKPOINT_FORMAT == "sqlite" and hasattr(self, "in_memory_cache"):
            # Si tenemos cache en memoria, usarlo para mejor rendimiento
            return self.in_memory_cache.get(address_str)
        else:
            # Para otros formatos o si no hay cache en memoria
            return self.address_to_city_cache.get(address_str)

    def _create_prompt(self, address):
        """
        Crea un prompt para extraer ciudades de direcciones.

        Args:
            address (str): Dirección a procesar

        Returns:
            str: Prompt formateado
        """
        return f"""
        Let's play a game. I will give you a string containing an address from anywhere in Japan. Your job is to extract and return ONLY the name of the city mentioned in the string.
This is the address: "{address}"

Rules:
1. Respond with ONLY the city name — no extra text, no explanations.
2. If you cannot confidently identify the city, respond with "Unknown".
3. If the input is empty or "EMPTY", respond with "Unknown".
4. Do NOT include postal codes, countries, regions, or other info.
5. If multiple cities are present, choose the most specific one.
6. Use the official local name of the city, with correct capitalization.
Your answer:
        """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception)),
        reraise=True,
    )
    def _call_openai_api(self, address):
        """
        Llama a la API de OpenAI con reintentos automáticos mediante tenacity.

        Args:
            address (str): Dirección a procesar

        Returns:
            str: Ciudad extraída
        """
        # Si está vacío, devolver Unknown directamente
        if not address or str(address).strip().upper() == "EMPTY":
            return "Unknown"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in geography who identifies city names in text.",
                    },
                    {"role": "user", "content": self._create_prompt(address)},
                ],
                temperature=0.0,
            )

            # Extraer respuesta
            city = response.choices[0].message.content.strip()
            return city

        except Exception as e:
            logger.warning(f"Error en llamada a API para '{address}': {str(e)}")
            raise  # Permite que tenacity maneje el reintento

    def _process_single_city(self, address):
        """
        Procesa una única dirección/ciudad con reintentos.

        Args:
            address (str): Dirección a procesar

        Returns:
            tuple: (dirección original, ciudad extraída)
        """
        # Convertir a string y limpiar
        address = str(address).strip()

        try:
            city = self._call_openai_api(address)
            return address, city
        except Exception as e:
            logger.error(
                f"No se pudo procesar '{address}' después de {self.max_retries} intentos: {str(e)}"
            )
            return address, "Unknown"

    def process_file(self, input_path, output_path, city_column):
        """
        Procesa un archivo usando ThreadPoolExecutor para concurrencia con procesamiento
        por lotes y guardado de checkpoints.

        Args:
            input_path (str): Ruta del archivo de entrada
            output_path (str): Ruta del archivo de salida
            city_column (str): Nombre de la columna que contiene las direcciones

        Returns:
            int: Número de direcciones únicas procesadas
        """
        try:
            # Cargar el archivo
            df = self._load_dataframe(input_path)
            # df_ya_procesados = df[df["STANDARDIZED_CITY"] != "Unknown"].copy()
            # df = df[df["STANDARDIZED_CITY"] == "Unknown"].copy()
            # Obtener direcciones únicas para procesar
            unique_addresses = df[city_column].dropna().unique()

            # Filtrar direcciones que ya están en el cache (solo si RESUME es True)
            if RESUME:
                addresses_to_process = [
                    addr
                    for addr in unique_addresses
                    if str(addr) not in self.address_to_city_cache
                ]
                logger.info(f"Modo RESUME: continuando desde checkpoint existente")
            else:
                addresses_to_process = unique_addresses
                logger.info(f"Modo NUEVO: procesando todas las direcciones")

            total_addresses = len(addresses_to_process)
            logger.info(f"Total direcciones únicas en archivo: {len(unique_addresses)}")

            if RESUME:
                logger.info(
                    f"Direcciones ya procesadas en cache: {len(unique_addresses) - total_addresses}"
                )

            logger.info(f"Direcciones pendientes de procesar: {total_addresses}")

            # Si hay direcciones para procesar
            if total_addresses > 0:
                # Crear barra de progreso
                with tqdm(
                    total=total_addresses,
                    desc=f"Procesando {os.path.basename(input_path)}",
                ) as pbar:
                    # Procesar por lotes
                    for i in range(0, total_addresses, self.batch_size):
                        batch = addresses_to_process[i : i + self.batch_size]
                        batch_size = len(batch)

                        logger.info(
                            f"Procesando lote {i // self.batch_size + 1}/{(total_addresses + self.batch_size - 1) // self.batch_size}: {batch_size} direcciones"
                        )

                        # Procesar direcciones usando ThreadPoolExecutor para concurrencia
                        with ThreadPoolExecutor(
                            max_workers=self.max_workers
                        ) as executor:
                            # Enviar tareas para este lote
                            future_to_address = {
                                executor.submit(
                                    self._process_single_city, address
                                ): address
                                for address in batch
                            }

                            # Procesar resultados a medida que se completan
                            for future in as_completed(future_to_address):
                                address, city = future.result()
                                self.address_to_city_cache[str(address)] = city
                                pbar.update(1)

                        # Guardar checkpoint después de cada lote
                        self._save_checkpoint(self.address_to_city_cache)

                        # Guardar un checkpoint parcial del DataFrame
                        if i + self.batch_size >= total_addresses or (
                            i > 0 and i % (self.batch_size * 5) == 0
                        ):
                            # Aplicar mapeo al DataFrame original con lo que ya tenemos
                            temp_df = df.copy()
                            temp_df["STANDARDIZED_CITY"] = temp_df[city_column].map(
                                lambda x: (
                                    self.address_to_city_cache.get(str(x), "Unknown")
                                    if pd.notna(x)
                                    else "Unknown"
                                )
                            )

                            # Guardar resultado intermedio
                            temp_output = output_path.replace(".", f"_partial_{i}.", 1)
                            self._save_dataframe(temp_df, temp_output)
                            logger.info(f"Guardado parcial en {temp_output}")
            else:
                logger.info("No hay nuevas direcciones para procesar")

            # Aplicar mapeo al DataFrame original con todos los datos procesados
            df["STANDARDIZED_CITY"] = df[city_column].map(
                lambda x: (
                    self.address_to_city_cache.get(str(x), "Unknown")
                    if pd.notna(x)
                    else "Unknown"
                )
            )
            standardized_mapping = {}

            # Para las direcciones únicas que ya has procesado
            for addr in unique_addresses:
                # Obtener el primer valor estandarizado para esta dirección
                standardized_value = df.loc[
                    df[city_column] == addr, "STANDARDIZED_CITY"
                ].iloc[0]
                standardized_mapping[addr] = standardized_value

            # Ahora aplica este mapeo a todas las filas del DataFrame original
            df["STANDARDIZED_CITY"] = df[city_column].map(standardized_mapping)
            # Unir con los ya procesados
            # df = pd.concat([df, df_ya_procesados], ignore_index=True)
            # Guardar resultado final
            self._save_dataframe(df, output_path)
            logger.info(f"Archivo guardado exitosamente en {output_path}")

            return len(unique_addresses)

        except Exception as e:
            logger.error(f"Error procesando archivo {input_path}: {str(e)}")
            # Intentar guardar lo que ya tenemos procesado
            self._save_checkpoint(self.address_to_city_cache)
            raise

    def _load_dataframe(self, file_path):
        """
        Carga un DataFrame desde un archivo CSV o Parquet.

        Args:
            file_path (str): Ruta del archivo

        Returns:
            pd.DataFrame: DataFrame cargado
        """
        if file_path.endswith(".parquet"):
            return pd.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Formato de archivo no soportado: {file_path}")

    def _save_dataframe(self, df, file_path):
        """
        Guarda un DataFrame en un archivo CSV o Parquet.

        Args:
            df (pd.DataFrame): DataFrame a guardar
            file_path (str): Ruta del archivo
        """
        # Asegurarse de que la carpeta existe
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if file_path.endswith(".parquet"):
            df.to_parquet(file_path, index=False)
        elif file_path.endswith(".csv"):
            df.to_csv(file_path, index=False)
        else:
            # Por defecto guardar como Parquet
            df.to_parquet(file_path, index=False)


def main():
    """Función principal"""
    # Mostrar configuración
    logger.info("Iniciando procesamiento con configuración:")
    logger.info(f"  - Carpeta: {FOLDER}")
    logger.info(f"  - Archivo: {FILE_NAME}")
    logger.info(f"  - Columna de ciudad: {CITY_COLUMN}")
    logger.info(f"  - Modelo: {MODEL}")
    logger.info(f"  - Tamaño de lote: {BATCH_SIZE}")
    logger.info(f"  - Trabajadores: {MAX_WORKERS}")
    logger.info(f"  - Formato checkpoint: {CHECKPOINT_FORMAT}")
    logger.info(f"  - Base checkpoint: {CHECKPOINT_FILE}")
    logger.info(f"  - Modo: {'Reanudar' if RESUME else 'Nuevo'}")

    # Preparar rutas
    input_file_path = os.path.join(FOLDER, FILE_NAME)
    file_name, file_ext = os.path.splitext(FILE_NAME)
    output_file_path = os.path.join(FOLDER, f"{file_name}-standardized2{file_ext}")

    # Determinar ruta de checkpoint según formato
    if CHECKPOINT_FORMAT == "sqlite":
        checkpoint_path = os.path.join(FOLDER, f"{CHECKPOINT_FILE}.db")
    elif CHECKPOINT_FORMAT == "pickle":
        checkpoint_path = os.path.join(FOLDER, f"{CHECKPOINT_FILE}.pkl")
    else:  # default to json
        checkpoint_path = os.path.join(FOLDER, f"{CHECKPOINT_FILE}.json")

    # Verificar que el archivo exista
    if not os.path.exists(input_file_path):
        logger.error(f"El archivo {input_file_path} no existe")
        return

    # Verificar si hay un archivo de salida existente cuando RESUME=False
    if not RESUME and os.path.exists(output_file_path):
        logger.warning(
            f"El archivo de salida {output_file_path} ya existe y RESUME=False"
        )
        respuesta = (
            input("¿Desea sobrescribir el archivo existente? (s/n): ").strip().lower()
        )
        if (
            respuesta != "s"
            and respuesta != "si"
            and respuesta != "y"
            and respuesta != "yes"
        ):
            logger.info("Operación cancelada por el usuario")
            return

        # Si se elige sobrescribir, también se elimina el checkpoint
        if os.path.exists(checkpoint_path) and not RESUME:
            logger.info(f"Eliminando checkpoint anterior: {checkpoint_path}")
            try:
                if CHECKPOINT_FORMAT == "sqlite":
                    # Para SQLite, solo eliminamos la tabla
                    conn = sqlite3.connect(checkpoint_path)
                    cursor = conn.cursor()
                    cursor.execute("DROP TABLE IF EXISTS cities")
                    conn.commit()
                    conn.close()
                    logger.info("Tabla de checkpoint SQLite eliminada")
                else:
                    # Para otros formatos, eliminamos el archivo
                    os.remove(checkpoint_path)
                    logger.info(f"Archivo de checkpoint eliminado: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Error al eliminar checkpoint: {str(e)}")

    # Tiempo de inicio
    start_time = time.time()

    try:
        # Inicializar validador de ciudades
        validator = CityValidator(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL,
            max_workers=MAX_WORKERS,
            max_retries=MAX_RETRIES,
            batch_size=BATCH_SIZE,
            checkpoint_file=CHECKPOINT_FILE,
        )

        # Procesar archivo
        cities_processed = validator.process_file(
            input_path=input_file_path,
            output_path=output_file_path,
            city_column=CITY_COLUMN,
        )

        # Mostrar estadísticas finales
        total_time = time.time() - start_time
        avg_time = total_time / cities_processed if cities_processed > 0 else 0

        logger.info("¡Procesamiento completado!")
        # Convertir segundos a horas, minutos y segundos
        total_hours = int(total_time // 3600)
        total_minutes = int((total_time % 3600) // 60)
        total_seconds = total_time % 60

        avg_hours = int(avg_time // 3600)
        avg_minutes = int((avg_time % 3600) // 60)
        avg_seconds = avg_time % 60

        logger.info(
            f"Tiempo total: {total_hours}h {total_minutes}m {total_seconds:.2f}s"
        )
        logger.info(
            f"Tiempo promedio por ciudad: {avg_hours}h {avg_minutes}m {avg_seconds:.4f}s"
        )

    except KeyboardInterrupt:
        logger.warning("Proceso interrumpido por el usuario")
        logger.info(
            "El progreso ha sido guardado en el checkpoint y puede continuarse más tarde."
        )
    except Exception as e:
        logger.error(f"Error en el procesamiento: {str(e)}")
    finally:
        # Mostrar tiempo total
        total_time = time.time() - start_time
        logger.info(f"Tiempo total de ejecución: {total_time:.2f} segundos")


if __name__ == "__main__":
    main()
