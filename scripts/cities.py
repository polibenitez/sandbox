import pandas as pd
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import random
import pandas as pd
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import random
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CityValidator:
   """
   Validador de ciudades usando la librería oficial de OpenAI con procesamiento
   concurrente para mayor eficiencia.
   """
   def __init__(self, api_key, base_url=None, model="gpt-3.5-turbo", max_workers=5, max_retries=3):
       """
       Inicializa el validador de ciudades.
      
       Args:
           api_key (str): Clave API de OpenAI
           base_url (str, optional): URL base para la API. Por defecto utiliza la de OpenAI.
           model (str): Modelo a utilizar. Por defecto "gpt-3.5-turbo".
           max_workers (int): Número máximo de hilos concurrentes.
           max_retries (int): Número máximo de reintentos para llamadas a la API.
       """
       self.model = model
       self.max_workers = max_workers
       self.max_retries = max_retries
      
       # Inicializar cliente de OpenAI
       client_args = {"api_key": api_key}
       if base_url:
           client_args["base_url"] = base_url
      
       self.client = OpenAI(**client_args)
      
   def _create_prompt(self, address):
       """
       Crea un prompt para extraer ciudades de direcciones.
      
       Args:
           address (str): Dirección a procesar
          
       Returns:
           str: Prompt formateado
       """
       return f"""Vamos a jugar un juego: yo te paso un string con direcciones de cualquier parte del continente europeo, y tú me tienes que dar SOLO el nombre de la ciudad que aparezca en ese string. Sólo responde con el nombre de la ciudad, sin añadir ningún otro texto o explicación.


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
                   {"role": "system", "content": "Eres un experto en geografía europea que identifica nombres de ciudades en textos."},
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
       Procesa un archivo usando ThreadPoolExecutor para concurrencia.
      
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
           total_addresses = len(unique_addresses)
           logger.info(f"Procesando {total_addresses} direcciones únicas en {input_path}")
          
           # Diccionario para almacenar resultados
           address_to_city = {}
          
           # Procesar direcciones usando ThreadPoolExecutor para concurrencia
           with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
               # Crear barra de progreso
               with tqdm(total=total_addresses, desc=f"Procesando {os.path.basename(input_path)}") as pbar:
                   # Enviar todas las tareas
                   future_to_address = {
                       executor.submit(self._process_single_city, address): address
                       for address in unique_addresses
                   }
                  
                   # Procesar resultados a medida que se completan
                   for future in as_completed(future_to_address):
                       address, city = future.result()
                       address_to_city[address] = city
                       pbar.update(1)
          
           # Aplicar mapeo al DataFrame original
           df['STANDARDIZED_CITY'] = df[city_column].map(address_to_city)
           df['STANDARDIZED_CITY'] = df['STANDARDIZED_CITY'].fillna("Unknown")
          
           # Guardar resultado
           self._save_dataframe(df, output_path)
           logger.info(f"Archivo guardado exitosamente en {output_path}")
          
           return total_addresses
          
       except Exception as e:
           logger.error(f"Error procesando archivo {input_path}: {str(e)}")
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
       if file_path.endswith('.parquet'):
           df.to_parquet(file_path, index=False)
       elif file_path.endswith('.csv'):
           df.to_csv(file_path, index=False)
       else:
           # Por defecto guardar como Parquet
           df.to_parquet(file_path, index=False)




class DataProcessor:
   """
   Clase para procesar múltiples archivos de datos de países.
   """
   def __init__(self, validator, folder, city_column):
       """
       Inicializa el procesador de datos.
      
       Args:
           validator (CityValidator): Validador de ciudades
           folder (str): Carpeta donde están los archivos
           city_column (str): Nombre de la columna que contiene las direcciones
       """
       self.validator = validator
       self.folder = folder
       self.city_column = city_column
  
   @staticmethod
   def find_country_files(folder_path, extension=None):
       """
       Busca archivos en la carpeta con la extensión especificada.
      
       Args:
           folder_path (str): Ruta de la carpeta
           extension (str, optional): Extensión de archivo
          
       Returns:
           list: Lista de archivos encontrados
       """
       files = []
      
       for filename in os.listdir(folder_path):
           file_path = os.path.join(folder_path, filename)
           if os.path.isfile(file_path):
               if extension is None or filename.endswith(extension):
                   files.append(filename)
      
       return files
  
   def process_countries(self, country_files=None, country_codes=None,
                        excluded_country_codes=None, extensions=None):
       """
       Procesa archivos de múltiples países.
      
       Args:
           country_files (list, optional): Lista específica de archivos a procesar
           country_codes (list, optional): Lista de códigos de país para filtrar archivos
           excluded_country_codes (list, optional): Lista de códigos de país a excluir
           extensions (list, optional): Lista de extensiones de archivo a procesar
          
       Returns:
           tuple: (archivos procesados, ciudades procesadas)
       """
       # Determinar qué archivos procesar
       if country_files is None:
           country_files = self._get_filtered_files(country_codes, excluded_country_codes, extensions)
      
       # Informar sobre los archivos que se van a procesar
       logger.info(f"Se procesarán {len(country_files)} archivos: {', '.join(country_files)}")
      
       # Estadísticas
       total_cities = 0
       total_files = 0
      
       # Crear barra de progreso para países
       with tqdm(total=len(country_files), desc="Progreso total", unit="países") as master_pbar:
           # Procesar cada archivo
           for file in country_files:
               country_code = file.split('.')[0]
               logger.info(f"Procesando archivo: {file} (País: {country_code})")
              
               # Rutas de archivos
               file_name, file_ext = os.path.splitext(file)
               input_file = os.path.join(self.folder, file)
               output_file = os.path.join(self.folder, f"{file_name}-standardized{file_ext}")
              
               # Tiempo de inicio
               start_time = time.time()
              
               # Procesar archivo
               cities_processed = self.validator.process_file(input_file, output_file, self.city_column)
              
               # Actualizar estadísticas
               total_cities += cities_processed
               total_files += 1
              
               # Tiempo de finalización
               elapsed_time = time.time() - start_time
               logger.info(f"Procesamiento de {file} completado. Ciudades: {cities_processed}. Tiempo: {elapsed_time:.2f}s")
              
               # Actualizar barra de progreso
               master_pbar.update(1)
      
       # Mostrar resumen
       logger.info("===== RESUMEN DE PROCESAMIENTO =====")
       logger.info(f"Total archivos procesados: {total_files}")
       logger.info(f"Total ciudades procesadas: {total_cities}")
      
       return total_files, total_cities
  
   def _get_filtered_files(self, country_codes=None, excluded_country_codes=None, extensions=None):
       """
       Obtiene archivos filtrados según criterios.
      
       Args:
           country_codes (list, optional): Lista de códigos de país para filtrar archivos
           excluded_country_codes (list, optional): Lista de códigos de país a excluir
           extensions (list, optional): Lista de extensiones de archivo a procesar
          
       Returns:
           list: Lista de archivos filtrados
       """
       all_files = []
       if extensions:
           for ext in extensions:
               all_files.extend(self.find_country_files(self.folder, ext))
       else:
           all_files = self.find_country_files(self.folder)
          
       # Filtrar por código de país
       if country_codes:
           country_files = [f for f in all_files if any(f.startswith(code.lower()) for code in country_codes)]
       else:
           country_files = all_files
      
       # Excluir países si es necesario
       if excluded_country_codes:
           country_files = [f for f in country_files
                           if not any(f.startswith(code.lower()) for code in excluded_country_codes)]
      
       return country_files




def main():
   """Función principal"""
   # Configuración
   API_KEY = "eyJhbGciOiP8"
   FOLDER = 'patents_complx_gb'
   CITY_COLUMN = "person_address"
   BASE_URL = "http://localhost:1234/v1"
   MODEL = "gemma-3-4b-it"
  
   # Países a procesar y excluir
   COUNTRY_CODES = ['gb1']
   EXCLUDED_COUNTRY_CODES = ['ES', 'IT', 'GR', 'CY']
  
   # Extensiones a procesar
   EXTENSIONS = ['.parquet']
  
   # Tiempo de inicio
   start_time = time.time()
  
   try:
       # Inicializar validador de ciudades
       validator = CityValidator(
           api_key=API_KEY,
           base_url=BASE_URL,
           model=MODEL,
           max_workers=2,
           max_retries=3
       )
      
       # Inicializar procesador de datos
       processor = DataProcessor(
           validator=validator,
           folder=FOLDER,
           city_column=CITY_COLUMN
       )
      
       # Procesar países
       files_processed, cities_processed = processor.process_countries(
           country_codes=COUNTRY_CODES,
           excluded_country_codes=EXCLUDED_COUNTRY_CODES,
           extensions=EXTENSIONS
       )
      
       # Mostrar estadísticas finales
       total_time = time.time() - start_time
       avg_time = total_time / cities_processed if cities_processed > 0 else 0
      
       logger.info(f"¡Procesamiento completado!")
       logger.info(f"Tiempo total: {total_time:.2f} segundos")
       logger.info(f"Tiempo promedio por ciudad: {avg_time:.4f} segundos")
      
   except KeyboardInterrupt:
       logger.warning("Proceso interrumpido por el usuario")
   except Exception as e:
       logger.error(f"Error en el procesamiento: {str(e)}")
   finally:
       # Mostrar tiempo total
       total_time = time.time() - start_time
       logger.info(f"Tiempo total de ejecución: {total_time:.2f} segundos")




if __name__ == "__main__":
   main()