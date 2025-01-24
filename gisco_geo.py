import pandas as pd
import requests
import time
from urllib.parse import quote
from tqdm import tqdm  # Para mostrar una barra de progreso

def build_url(row):
    """Construye la URL de la petición solo con los campos no nulos"""
    base_url = "https://gisco-services.ec.europa.eu/addressapi/search?"
    params = []
    
    if pd.notna(row['llm_country']):
        params.append(f"country={quote(str(row['llm_country']))}")
    if pd.notna(row['llm_city']):
        params.append(f"city={quote(str(row['llm_city']))}")
    if pd.notna(row['llm_postcode']):
        params.append(f"postcode={quote(str(row['llm_postcode']))}")
    
    return base_url + "&".join(params) if params else None

def get_gisco_data(url):
    """Realiza la petición a GISCO y devuelve N3 y XY si existe"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Levanta una excepción si el status code no es 200
        
        data = response.json()
        
        if data['count'] > 0:
            # Tomamos el primer resultado
            first_result = data['results'][0]
            return first_result.get('N3'), first_result.get('XY')
        
        return None, None
        
    except Exception as e:
        print(f"Error en la petición: {str(e)}")
        return None, None

def process_parquet_file(input_file, output_file, batch_size=100):
    """Procesa el archivo parquet por lotes"""
    # Leer el archivo parquet
    df = pd.read_parquet(input_file)
    
    # Añadir nuevas columnas
    df['n3_gisco'] = None
    df['geo_gisco'] = None
    
    # Procesar en lotes para evitar sobrecarga
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in tqdm(range(total_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]
        
        for idx, row in batch.iterrows():
            # Construir URL
            url = build_url(row)
            
            if url:
                # Realizar petición
                n3, xy = get_gisco_data(url)
                
                # Guardar resultados
                df.at[idx, 'n3_gisco'] = n3
                df.at[idx, 'geo_gisco'] = str(xy) if xy else None
                
            # Pequeña pausa para no sobrecargar la API
            time.sleep(0.1)
        
        # Guardar progreso cada 1000 registros
        if i % 10 == 0:
            df.to_parquet(output_file)
    
    # Guardar resultado final
    df.to_parquet(output_file)
    
    # Imprimir estadísticas
    total_processed = len(df)
    successful_matches = df['n3_gisco'].notna().sum()
    
    print(f"\nEstadísticas:")
    print(f"Total de registros procesados: {total_processed}")
    print(f"Coincidencias exitosas: {successful_matches}")
    print(f"Porcentaje de éxito: {(successful_matches/total_processed)*100:.2f}%")

if __name__ == "__main__":
    country = 'GR'
    folder = "patents_complx/data_by_country"
    input_file = f'{folder}/{country}/final.parquet'
    output_file = f'{folder}/{country}/final-gisco.parquet'
      
    process_parquet_file(input_file, output_file)