import pandas as pd
import oracledb
from datetime import datetime
import numpy as np
import os

LLAMA = ['SI' ]
GPT = []
PHI = ['BG','CZ', 'IT']

def get_oracle_type(dtype):
    if pd.api.types.is_numeric_dtype(dtype):
        return "NUMBER"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATE"
    else:
        return "VARCHAR2(4000)"

def load_parquet_to_oracle(parquet_path, schema_name, table_name, oracle_user, oracle_password, 
                          oracle_host, oracle_port, oracle_service_name, chunk_size=10000):
    try:
        dsn = (f"(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={oracle_host})"
               f"(PORT={oracle_port}))(CONNECT_DATA=(SERVICE_NAME={oracle_service_name})))")
        connection = oracledb.connect(user=oracle_user, password=oracle_password, dsn=dsn)
        cursor = connection.cursor()
        
        
        df = pd.read_parquet(parquet_path)
        #df = df.replace({float('nan'): None})
        # Crear la nueva columna 'llm_model' con base en los valores de 'PERSON_CTRY_CODE'
        df['llm_model'] = df['person_ctry_code'].apply(lambda x: 'llama33' if x in LLAMA else ('phi4' if x in PHI else 'GPT35'))
        # Replace all NaN, None, and numpy.nan with None
        df = df.replace([np.nan, 'None', 'NULL', ''], None)
        df = df.where(pd.notnull(df), None)

        # Create table if not exists
        #cursor.execute(f'DROP TABLE {schema_name}.{table_name}')
        columns = [f"{col} {get_oracle_type(df[col].dtype)}" for col in df.columns]
        create_table_sql = f"""
        CREATE TABLE {schema_name}.{table_name} (
            {', '.join(columns)}
        )"""
        drop_and_create_table_sql = f"""
        DROP TABLE IF EXISTS {schema_name}.{table_name};
        CREATE TABLE {schema_name}.{table_name} (
            {', '.join(columns)}
        )"""
        try:
            cursor.execute(create_table_sql)
        except Exception as e:
            print(f"Table creation error (might already exist): {str(e)}")
        
        # Handle datetime columns
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        columns = df.columns.tolist()
        bind_names = [f":{col}" for col in columns]
        print('inserting......')
        insert_sql = f"""
        INSERT INTO {schema_name}.{table_name}
        ({', '.join(columns)}) 
        VALUES ({', '.join(bind_names)})
        """
        #print(insert_sql)
        total_rows = len(df)
        start_time = datetime.now()
        
        for i in range(0, total_rows, chunk_size):
            chunk = df[i:i + chunk_size]
            #data = [dict(zip(columns, row)) for row in chunk.values.tolist()]
            data = [tuple(None if pd.isna(x) else x for x in row) for row in chunk.values]
            #print(data)
            cursor.executemany(insert_sql, data)
            connection.commit()
            print(f"Processed {min(i + chunk_size, total_rows)}/{total_rows} rows")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        if 'connection' in locals():
            connection.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

def find_data_to_save_parquet_files(root_folder):
    """
    Searches for all 'remaining.parquet' files in the given folder and its subdirectories.

    Args:
        root_folder (str): The path to the main folder where to start searching.

    Returns:
        list: A list of file paths for 'remaining.parquet' files found.
    
    remaining_parquet_files = []

    # Walk through the folder and its subfolders
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # If 'remaining.parquet' is in the current directory, add it to the list
        if 'data_to_save.parquet' in filenames:
            full_path = os.path.join(dirpath, 'data_to_save.parquet')
            remaining_parquet_files.append(full_path)

    return remaining_parquet_files"""

def find_data_to_save_parquet_files(folder_path):
    """
    Busca recursivamente todos los archivos 'final.parquet' dentro del directorio especificado
    y sus subdirectorios.
    
    Args:
        folder_path (str): Ruta del directorio donde buscar
        
    Returns:
        list: Lista con las rutas completas de los archivos final.parquet encontrados
    """
    parquet_files = []
    
    # Recorremos el directorio y subdirectorios
    for root, dirs, files in os.walk(folder_path):
        # Buscamos archivos final.parquet en el directorio actual
        if 'final.parquet' in files:
            # Construimos la ruta completa del archivo
            file_path = os.path.join(root, 'final.parquet')
            # Añadimos la ruta a la lista
            parquet_files.append(file_path)
            
    return parquet_files

if __name__ == "__main__":
    # Example usage
    files = find_data_to_save_parquet_files('patents_complx/data_by_country_2/')
    for file in files:
        print(f'Load file - {file}')
        config = {
            'parquet_path': file,
            'schema_name': 'PROJECT_COMPLEXITY',
            'table_name': 'PATENTS_LAT_LON',
            'oracle_user': 'PROJECT_COMPLEXITY',
            'oracle_password': 'scu35Vd.E2',
            'oracle_host': 'siprodb01p-dars.jrc.it',
            'oracle_port': 1525,
            'oracle_service_name': 'SCIDB',
            'chunk_size': 10000
        }
        
        load_parquet_to_oracle(**config)
        print(f'File loaded- {file}')
