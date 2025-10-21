import pandas as pd
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import random
import json
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ==== CONFIGURACIÓN GLOBAL ====
# Configuración de API
API_KEY = "eyJhbGciCEB_k"  # Clave API de OpenAI
BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "gemma-3-4b-it-qat"

# Configuración de procesamiento
FOLDER = "data/us/"  # Carpeta donde está el archivo y donde se guardarán los resultados
FILE_NAME = "part.0.state.parquet"  # Nombre del archivo a procesar
CITY_COLUMN = "person_address"  # Nombre de la columna que contiene las direcciones
BATCH_SIZE = 50  # Tamaño de lote para procesamiento
MAX_WORKERS = 3  # Número máximo de hilos concurrentes
MAX_RETRIES = 3  # Número máximo de reintentos

# Configuración de archivos
CHECKPOINT_FILE = "city_checkpoint.json"  # Archivo para guardar el progreso por ciudades

# Modo de ejecución
RESUME = True  # Si es True, reanuda desde el último checkpoint

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CityValidator:
    """
    Validador de ciudades usando la librería oficial de OpenAI con procesamiento
    concurrente para mayor eficiencia.
    """
    def __init__(self, api_key, base_url=None, model="llama-3.3-70b-instruct", max_workers=5, max_retries=3, 
                 batch_size=100, checkpoint_file=None):
        self.model = model
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.batch_size = batch_size
        
        # Establecer ruta completa del checkpoint
        if checkpoint_file:
            self.checkpoint_file = os.path.join(FOLDER, checkpoint_file)
        else:
            self.checkpoint_file = os.path.join(FOLDER, CHECKPOINT_FILE)
        
        # Inicializar cliente de OpenAI
        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url
        
        self.client = OpenAI(**client_args)
        
        # Cargar el checkpoint si existe
        self.address_to_city_cache = self._load_checkpoint()
    
    def _load_checkpoint(self):
        """
        Carga el checkpoint si existe.
        
        Returns:
            dict: Diccionario con los datos cacheados (address -> city)
        """
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    logger.info(f"Cargando checkpoint desde {self.checkpoint_file}")
                    return json.load(f)
            else:
                logger.info("No se encontró checkpoint previo")
                return {}
        except Exception as e:
            logger.warning(f"Error al cargar checkpoint: {str(e)}. Continuando sin datos cacheados.")
            return {}
    
    def _save_checkpoint(self, data):
        """
        Guarda el checkpoint con los datos procesados.
        
        Args:
            data (dict): Diccionario con los datos a guardar (address -> city)
        """
        try:
            # Asegurarse de que la carpeta existe
            os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
            
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"Checkpoint guardado en {self.checkpoint_file}")
        except Exception as e:
            logger.error(f"Error al guardar checkpoint: {str(e)}")
    
    def _create_prompt(self, address):
        """
        Crea un prompt para extraer ciudades de direcciones.
        
        Args:
            address (str): Dirección a procesar
            
        Returns:
            str: Prompt formateado
        """
        english_prompt = f"""
        Let's play a game. I will give you a string containing an address from anywhere in Europe. Your job is to extract and return ONLY the name of the city mentioned in the string.
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
        
        spanish_prompt =  f"""Vamos a jugar un juego: yo te paso un string con direcciones de cualquier parte del continente europeo, y tú me tienes que dar SOLO el nombre de la ciudad que aparezca en ese string. Sólo responde con el nombre de la ciudad, sin añadir ningún otro texto o explicación.

Este es el texto: "{address}"

Reglas:
1. Responde ÚNICAMENTE con el nombre de la ciudad, nada más.
2. Si no puedes determinar la ciudad con seguridad, responde solo "Unknown".
3. Si el texto está vacío o es "EMPTY", responde solo "Unknown".
4. No incluyas códigos postales, países, regiones ni otra información.
5. Si hay varias posibles ciudades, elige la más específica.
6. Usa el nombre local oficial de la ciudad (no su versión en inglés), respetando mayúsculas y minúsculas adecuadamente.

Recuerda: responde SOLO con el nombre de la ciudad o "Unknown".

Tu respuesta:"""
        return english_prompt

    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception)),
        reraise=True
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
                    {"role": "system", "content": "You are an expert in European geography who identifies city names in text."},
                    {"role": "user", "content": self._create_prompt(address)}
                ],
                temperature=0.0
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
            logger.error(f"No se pudo procesar '{address}' después de {self.max_retries} intentos: {str(e)}")
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
            
            # Obtener direcciones únicas para procesar
            unique_addresses = df[city_column].dropna().unique()
            
            # Filtrar direcciones que ya están en el cache
            addresses_to_process = [addr for addr in unique_addresses if str(addr) not in self.address_to_city_cache]
            
            total_addresses = len(addresses_to_process)
            logger.info(f"Total direcciones únicas: {len(unique_addresses)}")
            logger.info(f"Direcciones ya procesadas en cache: {len(unique_addresses) - total_addresses}")
            logger.info(f"Direcciones pendientes de procesar: {total_addresses}")
            
            # Si hay direcciones para procesar
            if total_addresses > 0:
                # Crear barra de progreso
                with tqdm(total=total_addresses, desc=f"Procesando {os.path.basename(input_path)}") as pbar:
                    # Procesar por lotes
                    for i in range(0, total_addresses, self.batch_size):
                        batch = addresses_to_process[i:i + self.batch_size]
                        batch_size = len(batch)
                        
                        logger.info(f"Procesando lote {i // self.batch_size + 1}/{(total_addresses + self.batch_size - 1) // self.batch_size}: {batch_size} direcciones")
                        
                        # Procesar direcciones usando ThreadPoolExecutor para concurrencia
                        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                            # Enviar tareas para este lote
                            future_to_address = {
                                executor.submit(self._process_single_city, address): address 
                                for address in batch
                            }
                            
                            # Procesar resultados a medida que se completan
                            for future in as_completed(future_to_address):
                                address, city = future.result()
                                self.address_to_city_cache[str(address)] = city
                                pbar.update(1)
                        
                        # Guardar checkpoint después de cada lote
                        self._save_checkpoint(self.address_to_city_cache)
                        
                        # Opcionalmente, también podríamos guardar un checkpoint parcial del DataFrame
                        if i + self.batch_size >= total_addresses or (i > 0 and i % (self.batch_size * 5) == 0):
                            # Aplicar mapeo al DataFrame original con lo que ya tenemos
                            temp_df = df.copy()
                            temp_df['STANDARDIZED_CITY'] = temp_df[city_column].map(
                                lambda x: self.address_to_city_cache.get(str(x), "Unknown")
                            )
                            
                            # Guardar resultado intermedio
                            temp_output = output_path.replace('.', f'temp/_partial_{i}.', 1)
                            self._save_dataframe(temp_df, temp_output)
                            logger.info(f"Guardado parcial en {temp_output}")
            
            # Aplicar mapeo al DataFrame original con todos los datos procesados
            df['STANDARDIZED_CITY'] = df[city_column].map(
                lambda x: self.address_to_city_cache.get(str(x), "Unknown") if pd.notna(x) else "Unknown"
            )
            
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
        if file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
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
        
        if file_path.endswith('.parquet'):
            df.to_parquet(file_path, index=False)
        elif file_path.endswith('.csv'):
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
    logger.info(f"  - Checkpoint: {CHECKPOINT_FILE}")
    logger.info(f"  - Modo: {'Reanudar' if RESUME else 'Nuevo'}")
    
    # Preparar rutas
    input_file_path = os.path.join(FOLDER, FILE_NAME)
    file_name, file_ext = os.path.splitext(FILE_NAME)
    output_file_path = os.path.join(FOLDER, f"{file_name}-standardized{file_ext}")
    
    # Verificar que el archivo exista
    if not os.path.exists(input_file_path):
        logger.error(f"El archivo {input_file_path} no existe")
        return
    
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
            checkpoint_file=CHECKPOINT_FILE
        )
        
        # Procesar archivo
        cities_processed = validator.process_file(
            input_path=input_file_path,
            output_path=output_file_path,
            city_column=CITY_COLUMN
        )
        
        # Mostrar estadísticas finales
        total_time = time.time() - start_time
        avg_time = total_time / cities_processed if cities_processed > 0 else 0
        
        logger.info(f"¡Procesamiento completado!")
        logger.info(f"Tiempo total: {total_time:.2f} segundos")
        logger.info(f"Tiempo promedio por ciudad: {avg_time:.4f} segundos")
        
    except KeyboardInterrupt:
        logger.warning("Proceso interrumpido por el usuario")
        logger.info("El progreso ha sido guardado en el checkpoint y puede continuarse más tarde.")
    except Exception as e:
        logger.error(f"Error en el procesamiento: {str(e)}")
    finally:
        # Mostrar tiempo total
        total_time = time.time() - start_time
        logger.info(f"Tiempo total de ejecución: {total_time:.2f} segundos")


if __name__ == "__main__":
    main()