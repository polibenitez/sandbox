import pandas as pd
import pyarrow.parquet as pq
from openai import OpenAI
import time
from typing import List, Dict
import logging
from tqdm import tqdm
import os

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AddressProcessor:
    def __init__(self, api_key: str, base_url: str, model: str, batch_size: int = 20, max_retries: int = 3):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.model = model
        
    def _create_prompt(self, address: str) -> str:
        limpiar_ciudades  = f"""Given the city name "{address}", what is its standard English name? 
        If you can identify the city despite typos or being in a different language, 
        return its common English name. If you cannot determine the city with confidence, 
        return exactly "Unknown". Only return the city name, nothing else."""


        extraer_ciudad = f"""Extract only the city name from this address. Return ONLY the city name, nothing else.
        Focus on identifying city names, you can use the posta code if it is in the address to identify the city. If you cannot identify a city for an address or if the city name is ambiguous or incomplete, return 'Unknown'.
        
        Address: {address}
        """
        extraer_ciudad_cp = f"""Extract the city name and the post code from this address. Return ONLY the city name and post code, nothing else.
        Focus on identifying city names and postcodes, you can use the post code if it is in the address to identify the city. If you cannot identify a city or the post code, return 'Unknown'.
        
        Address: {address}
        """
        return extraer_ciudad
    
    def _process_batch(self, addresses: List[str]) -> List[str]:
        """Procesa un lote de direcciones usando la API de OpenAI"""
        try:
            # Crear los mensajes para cada dirección en el lote
            messages = [
                [{"role": "system", "content": "You are a helpful assistant that identifies city names and provides their standard English names."},
                 {"role": "user", "content": self._create_prompt(addr)}]
                for addr in addresses
            ]
            
            # Realizar la llamada a la API en batch
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages[0],  # Solo necesitamos enviar un mensaje
                n=len(addresses),  # Número de completions que queremos
                temperature=0.0  # Mantener las respuestas consistentes
            )
            
            # Extraer las ciudades de las respuestas
            cities = [choice.message.content.strip() for choice in response.choices]
            return cities
            
        except Exception as e:
            logger.error(f"Error procesando batch: {str(e)}")
            return ["Unknown"] * len(addresses)

    def process_file(self, input_path: str, output_path: str, address_column: str):
        """Procesa el archivo parquet completo"""
        try:
            # Leer el archivo parquet
            df = pd.read_parquet(input_path)
            # Leer el archivo CSV
            #df = pd.read_csv('citiestobetranslated.csv')
            
            # Obtener todas las direcciones únicas para evitar procesamiento duplicado
            unique_addresses = df[address_column].unique()
            logger.info(f"Procesando {len(unique_addresses)} direcciones únicas")
            
            # Diccionario para almacenar el mapeo dirección -> ciudad
            address_to_city = {}
            
            # Procesar las direcciones en lotes con barra de progreso
            for i in tqdm(range(0, len(unique_addresses), self.batch_size)):
                batch = unique_addresses[i:i + self.batch_size]
                
                # Intentar procesar el lote con reintentos
                for attempt in range(self.max_retries):
                    try:
                        cities = self._process_batch(batch)
                        # Actualizar el diccionario de mapeo
                        address_to_city.update(dict(zip(batch, cities)))
                        break
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            logger.error(f"Error después de {self.max_retries} intentos: {str(e)}")
                            # En caso de error, marcar como Unknown
                            address_to_city.update(dict(zip(batch, ["Unknown"] * len(batch))))
                        else:
                            time.sleep(2 ** attempt)  # Backoff exponencial
                
            # Aplicar el mapeo al DataFrame original
            df['ciudad'] = df[address_column].map(address_to_city)
            
            # Guardar el resultado
            df.to_parquet(output_path, index=False)
            logger.info(f"Archivo guardado exitosamente en {output_path}")
            
        except Exception as e:
            logger.error(f"Error procesando el archivo: {str(e)}")
            raise


def find_country_parquet_files(folder_path):
    """
    Retrieves all parquet files named as country codes (XX.parquet) from the specified folder.
    This function does not search in subdirectories.

    Args:
        folder_path (str): The path to the folder where to search for the parquet files.

    Returns:
        list: A list of file paths for all country-based parquet files (XX.parquet).
    """
    country_parquet_files = []
    
    # Get the list of all files in the folder (excluding subfolders)
    for filename in os.listdir(folder_path):
        # Check if the filename matches the pattern: XX.parquet (where XX is two uppercase letters)
        if filename.endswith('.parquet'):
            # Add the full path to the list
            country_parquet_files.append(filename)
    
    return country_parquet_files



def main():
    # Configuración
    API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY4YjEwY2JiLWY2OGMtNGYxYS04YWMwLWY5N2YzOWU3MTI1ZiIsImlzcyI6ImdwdGpyYyIsImlhdCI6MTczOTI3MTY4MCwiZXhwIjoxNzU2NTk4NDAwLCJpc19yZXZva2VkIjpmYWxzZSwiYWNjb3VudF9pZCI6IjAwMzJlNDQ4LTBlNTctNGQ1Mi05M2NjLTcwMTJhOTI5ZTgzOSIsInVzZXJuYW1lIjoibWFudWVsLmJlbml0ZXotc2FuY2hlekBleHQuZWMuZXVyb3BhLmV1IiwicHJvamVjdF9pZCI6IklJRCIsImRlcGFydG1lbnQiOiJKUkMuVC42IiwicXVvdGFzIjpbeyJtb2RlbF9uYW1lIjoiZ3B0LTRvIiwiZXhwaXJhdGlvbl9mcmVxdWVuY3kiOiJkYWlseSIsInZhbHVlIjo0MDAwMDB9LHsibW9kZWxfbmFtZSI6ImdwdC0zNS10dXJiby0xMTA2IiwiZXhwaXJhdGlvbl9mcmVxdWVuY3kiOiJkYWlseSIsInZhbHVlIjo0MDAwMDB9LHsibW9kZWxfbmFtZSI6ImdwdC00LTExMDYiLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH1dLCJhY2Nlc3NfZ3JvdXBzIjpbeyJpZCI6Ijk5Mjc5OWY1LTZiZGUtNGJjMS05NDZjLTc0NDU4MTFmMDJjMCIsImFjY2Vzc19ncm91cCI6ImdlbmVyYWwifV19.Dmq3xbyl4AjLpAFgSkjACG6dB8ZZwC2qecSpLs6O7rQ"
    FOLDER = 'postcodes'
    INPUT_FILE = 'citiestobetranslated.csv'
    OUTPUT_FILE = 'citiestranslated-final.csv'
    ADDRESS_COLUMN = "PERSON_ADDRESS"
    BATCH_SIZE = 20
    BASE_URL = "https://api-gpt.jrc.ec.europa.eu/v1"
    MODEL = "llama-3.1-8b-instruct"
    #MODEL = "gpt-4o"
    BASE_OLLAMA = "http://localhost:11434/v1"
    MODEL_OLLAMA = "deepseek-r1:1.5b"
    country_files = find_country_parquet_files(FOLDER)
    # Inicializar el procesador
    processor = AddressProcessor(
        api_key=API_KEY,
        base_url=BASE_URL,
        model = MODEL,
        batch_size=BATCH_SIZE
    )
    countries_included = ['ES-final-filtrado.parquet','FR-final-filtrado.parquet','IT-final-filtrado.parquet']
    process_countries(FOLDER, ADDRESS_COLUMN, country_files, processor, countries_included)
    #processor.process_file(INPUT_FILE, OUTPUT_FILE, 'PERSON_CITY')
    print("Process completed and saved")

def process_countries(FOLDER, ADDRESS_COLUMN, country_files, processor: AddressProcessor, countries_included):
    for country in country_files:
        if country not in countries_included:
            print(f'Procesing {country} file')
            # Inicia el contador de tiempo para este archivo
            start_time_file = time.time()
            # Procesar el archivo
            # Extraer el nombre base del archivo (sin extensión)
            file_name = os.path.splitext(os.path.basename(country))[0]

            # Crear el nombre del archivo de salida
            OUTPUT_FILE = f'{FOLDER}/{file_name}-final.parquet'
            INPUT_FILE = f'{FOLDER}/{country}'
            processor.process_file(INPUT_FILE, OUTPUT_FILE, ADDRESS_COLUMN)

            # Finaliza el contador de tiempo para este archivo
            end_time_file = time.time()
            elapsed_time_file = end_time_file - start_time_file
            
            print(f'Process finished for file {country}. Time elapsed: {elapsed_time_file:.2f} seconds')
    

if __name__ == "__main__":
    main()


