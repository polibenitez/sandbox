import pandas as pd
from openai import OpenAI
import json
import re
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from pathlib import Path
from datetime import datetime
from openai import RateLimitError
import os
import time
from datetime import timedelta
from tqdm import tqdm  # Importar tqdm para la barra de progreso

TOKEN = "eyJhbG"
#BASE_URL = "http://127.0.0.1:1234/v1"
BASE_URL = "http://localhost:11434/v1"
MAX_WORKERS = 5
BATCH_SIZE = 100 
#MODEL = 'deepseek-r1-distill-qwen-1.5b'
MODEL = "deepseek-r1:1.5b"
FOLDER = 'patents_complx_invt'

def create_output_directory(file_path):
    """Create output directory based on parquet file name"""
    file_name = Path(file_path).stem  # Gets filename without extension
    output_dir = Path(FOLDER) / file_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_processed_df(original_df, new_columns):
    """
    Helper function to create DataFrame with processed records
    """
    # Get minimum length across all new columns
    min_length = min(len(values) for values in new_columns.values())
    
    # Create DataFrame with processed rows
    processed_df = original_df.iloc[:min_length].copy()
    
    # Add processed columns
    for col, values in new_columns.items():
        processed_df[col] = values[:min_length]
    
    # Remove rows where all LLM columns are None
    llm_columns = list(new_columns.keys())
    processed_df = processed_df.dropna(subset=llm_columns, how='all')
    
    # Verify columns exist and have values
    for col in llm_columns:
        if col not in processed_df.columns:
            print(f"Warning: Column {col} missing from DataFrame")
        else:
            print(f"Column {col} has {processed_df[col].count()} non-null values")
    
    return processed_df

def safe_save_dataframe(df, output_path, prefix=''):
    """Save DataFrame with verification"""
    print(f"\nAttempting to save DataFrame with {len(df)} records")
    print(f"DataFrame info:")
    print(df.info())
    parquet_path = output_path / f'{prefix}.parquet'
    csv_path = output_path / f'{prefix}.csv'
    
    print("\nColumns before saving:", df.columns.tolist())
    print("Sample of data before saving:")
    print(df.head())
    
    try:
        df.to_parquet(parquet_path)
        print(f"Saved {len(df)} records to {parquet_path}")
        # Verify saved data
        test_read = pd.read_parquet(parquet_path)
        print("Successfully verified saved data")
        return True, parquet_path
    except Exception as e:
        print(f"Parquet save error: {str(e)}")
        try:
            df.to_csv(csv_path, index=False)
            print(f"Saved to CSV: {csv_path}")
            return True, csv_path
        except Exception as csv_error:
            print(f"CSV save error: {str(csv_error)}")
            return False, None

def parse_llama_response(response_string):
    try:
        # Find the JSON array pattern in the string
        pattern = r'\[(.*?)\]'
        match = re.search(pattern, response_string, re.DOTALL)
        
        if match:
            # Extract the JSON string and parse it
            json_str = '[' + match.group(1) + ']'
            return json.loads(json_str)
        else:
            return None
            
    except json.JSONDecodeError as e:
        print(response_string)
        print(f"Error parsing JSON: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def get_last_processed_record(output_dir: Path) -> Tuple[int, pd.DataFrame]:
    """
    Find the last successfully processed record by examining existing output files.
    Returns the index to resume from and the last processed dataframe.
    """
    max_processed_index = -1
    last_processed_df = None
    
    # Get all parquet files in the directory
    parquet_files = list(output_dir.glob('*.parquet'))
    
    for file in parquet_files:
        try:
            # Extract index from checkpoint files
            if 'checkpoint_' in file.name:
                index = int(re.search(r'checkpoint_(\d+)', file.name).group(1))
                if index > max_processed_index:
                    max_processed_index = index
                    last_processed_df = pd.read_parquet(file)
            
            # Check ratelimit files
            elif 'ratelimit_' in file.name:
                index = int(re.search(r'ratelimit_(\d+)', file.name).group(1))
                if index > max_processed_index:
                    max_processed_index = index
                    last_processed_df = pd.read_parquet(file)
        except Exception as e:
            print(f"Error reading file {file}: {str(e)}")
            continue
    
    return max_processed_index, last_processed_df

def save_checkpoint(df, new_columns, start_index, output_path, previous_records=0, suffix='checkpoint'):
    """
    Helper function to save processing checkpoints with proper length handling
    """
    # Usar get_processed_df para asegurar que las longitudes coincidan
    temp_df = get_processed_df(df, new_columns)
    
    # Add timestamp to checkpoint filename to avoid overwrites
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    total_records = len(temp_df) + previous_records
    checkpoint_path = output_path / f'{suffix}_{total_records}_{timestamp}.parquet'
    temp_df.to_parquet(checkpoint_path)
    return temp_df

def get_address_from_llm(address: List):
    global client
    while True:
        try:
            client = OpenAI(
                api_key=TOKEN,
                base_url=BASE_URL
            )
            
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": """
                        Extract the city name from each of the following 9 different addresses. 
                     Focus on identifying city names that are explicitly mentioned. If you cannot identify a city for an address or if the city name is ambiguous or incomplete, respond with only the word EMPTY. 
                     Provide the results coma separated in the same order as the addresses given. Do not add any comment, just reply with the name of the city or EMPTY:
                    """},
                    {"role": "user", "content": str(address)}
                ],
                temperature=0
            )
            sleep(0.5)
            llm_response = response.choices[0].message.content
            return parse_llama_response(llm_response)
            
        except RateLimitError as e:
            print(e)
            print(f"Rate limit exceeded for token ending in ...")
                
        except Exception as e:
            print(f"Error processing address {address}: {str(e)}")
            return []

def process_parquet_in_batches(df, output_path, batch_size=10, checkpoint_frequency=500, previous_records=0):
    new_columns = {
        'llm_street': [],
        'llm_city': [],
        'llm_postcode': [],
        'llm_country': [],
        'llm_nuts3': []
    }
    
    total_processed = 0
    batch_processed = 0
    
    try:
        for start in tqdm(range(0, len(df), batch_size), desc="Processing records"):
            try:
                end = start + batch_size
                batch = df.iloc[start:end]
                current_batch_size = len(batch)
                
                address_list = get_address_from_llm(batch['person_address'].tolist())
                print(address_list)
                if address_list:
                    for addr in address_list:
                        if addr:
                            for col in new_columns:
                                field = col.replace('llm_', '')
                                new_columns[col].append(addr.get(field))
                            batch_processed += 1
                        else:
                            for col in new_columns:
                                new_columns[col].append(None)
                    
                    if start % checkpoint_frequency == 0 and new_columns['llm_street']:
                        processed_df = get_processed_df(df.iloc[:start + current_batch_size], new_columns)
                        total_processed = len(processed_df)
                        
                        save_checkpoint(
                            processed_df,
                            new_columns,
                            start,
                            output_path,
                            previous_records=previous_records
                        )
                        print(f"Checkpoint saved. Total processed records: {total_processed + previous_records}")
                
            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected in batch processing. Saving progress...")
                if new_columns['llm_street']:
                    processed_df = get_processed_df(df.iloc[:start + current_batch_size], new_columns)
                    save_checkpoint(
                        processed_df,
                        new_columns,
                        start,
                        output_path,
                        previous_records=previous_records,
                        suffix='interrupt'
                    )
                raise
                
            except RateLimitError:
                if new_columns['llm_street']:
                    processed_df = get_processed_df(df.iloc[:start + current_batch_size], new_columns)
                    return processed_df, "ratelimit"
                return None, "ratelimit"
                
            except Exception as e:
                print(f"Error in batch starting at {start}: {str(e)}")
                continue
        
        if new_columns['llm_street']:
            final_processed_df = get_processed_df(df, new_columns)
            return final_processed_df, "success"
        
        return None, "error"
        
    except Exception as e:
        print(f"Uncaught exception: {str(e)}")
        if new_columns['llm_street']:
            processed_df = get_processed_df(df, new_columns)
            return processed_df, "error"
        return None, "error"

def process_parquet_file(file_name):
    file_path = Path(FOLDER) / file_name
    final_df = None
    
    try:
        output_dir = create_output_directory(file_path)
        print(f"Created/accessed output directory: {output_dir}")
        
        start_index, last_df = get_last_processed_record(output_dir)
        previous_records = len(last_df) if last_df is not None else 0
        
        df = pd.read_parquet(file_path)
        
        if start_index >= 0:
            print(f"Resuming from index {start_index}")
            remaining_df = df.iloc[start_index:]
            processed_df, status = process_parquet_in_batches(
                remaining_df, 
                output_dir,
                previous_records=previous_records
            )
            
            if processed_df is not None and last_df is not None:
                final_df = pd.concat([last_df, processed_df], ignore_index=True)
            else:
                final_df = processed_df or last_df
        else:
            print("Starting from beginning")
            final_df, status = process_parquet_in_batches(df, output_dir)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Saving current progress...")
        if final_df is not None and not final_df.empty:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            success, path = safe_save_dataframe(
                final_df, 
                output_dir, 
                f'interrupt_{timestamp}'
            )
            if success:
                print(f"Successfully saved progress to {path}")
            else:
                print("Failed to save progress")
        raise  # Re-raise the interrupt to properly exit the script
        
    except Exception as e:
        print(f"Error in process_parquet_file: {str(e)}")
        if final_df is not None and not final_df.empty:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_save_dataframe(final_df, output_dir, f'error_recovery_{timestamp}')
    
    finally:
        if final_df is not None and not final_df.empty:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_save_dataframe(final_df, output_dir, f'final_{timestamp}')

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
        if filename.endswith('.parquet') and len(filename) == 10:
            # Add the full path to the list
            country_parquet_files.append(filename)
    
    return country_parquet_files

# Función para convertir segundos a horas, minutos y segundos
def format_time(seconds):
    hours = int(seconds // 3600)  # 1 hora = 3600 segundos
    seconds %= 3600
    minutes = int(seconds // 60)  # 1 minuto = 60 segundos
    seconds %= 60
    return f"{hours}h {minutes}m {seconds:.2f}s"

country_files = find_country_parquet_files(FOLDER)
countries_excluded = ['CY.parquet', 'EE.parquet', 'HR.parquet', 'LT.parquet', 'LV.parquet','MT.parquet','SK.parquet', 'PL.parquet', 'GR.parquet', 'ES.parquet', 'SE.parquet', 'RO.parquet', 'AT.parquet','BE.parquet', 'FI.parquet', 'DK.parquet', 'BG.parquet', 'IT.parquet', 'SI.parquet', 'CZ.parquet','HU.parquet']
countries_included = ['SK.parquet']

for country in country_files:
    if country in countries_included:
        print(f'Procesing {country} file')
        # Inicia el contador de tiempo para este archivo
        start_time_file = time.time()
        process_parquet_file(country)
        # Finaliza el contador de tiempo para este archivo
        end_time_file = time.time()
        elapsed_time_file = end_time_file - start_time_file
        
        print(f'Process finished for file {country}. Time elapsed: {format_time(elapsed_time_file)} seconds')
print("Process completed and saved")