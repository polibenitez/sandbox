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
from datetime import timedelta


TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjQ3NGQ5MDZkLTc0MDQtNGJkMS04OThmLTVlMzMyY2NmNzM3ZiIsImlzcyI6ImdwdGpyYyIsImlhdCI6MTcwMzI4ODc3OSwiZXhwIjoxNzQ4NjQ5NjAwLCJpc19yZXZva2VkIjpmYWxzZSwiYWNjb3VudF9pZCI6ImJkMzMzYjk0LTMzYTAtNDc5MS04YjY2LTJlYWFjMmZiMzgwNiIsInVzZXJuYW1lIjoiVmljdG9yaWFuby5NQURST05BTC1ERS1MT1MtU0FOVE9TQGV4dC5lYy5ldXJvcGEuZXUiLCJwcm9qZWN0X2lkIjoiSW5kdXN0cmlhbF9Jbm5vdmF0aW9uX2FuZF9EeW5hbWljcyIsInF1b3RhcyI6W3sibW9kZWxfbmFtZSI6ImdwdC0zNS10dXJiby0wMzAxIiwiZXhwaXJhdGlvbl9mcmVxdWVuY3kiOiJkYWlseSIsInZhbHVlIjo0MDAwMDB9LHsibW9kZWxfbmFtZSI6ImdwdC0zNS10dXJiby0wNjEzIiwiZXhwaXJhdGlvbl9mcmVxdWVuY3kiOiJkYWlseSIsInZhbHVlIjo0MDAwMDB9LHsibW9kZWxfbmFtZSI6ImdwdC0zNS10dXJiby0xNmsiLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH0seyJtb2RlbF9uYW1lIjoiZ3B0LTQiLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH0seyJtb2RlbF9uYW1lIjoiZ3B0LTQtMzJrIiwiZXhwaXJhdGlvbl9mcmVxdWVuY3kiOiJkYWlseSIsInZhbHVlIjo0MDAwMDB9LHsibW9kZWxfbmFtZSI6ImdwdC0zNS10dXJiby0xMTA2IiwiZXhwaXJhdGlvbl9mcmVxdWVuY3kiOiJkYWlseSIsInZhbHVlIjo0MDAwMDB9LHsibW9kZWxfbmFtZSI6ImdwdC00LTExMDYiLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH1dfQ.SRoH8vMr1zNdmsjtG3W604wkaomxEjCDdkMPkrmls3w"
MY_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjBiMzU0YTdjLTkyZGMtNDMyYS05ZjA4LWQ0NGZmNmFlMDhjNyIsImlzcyI6ImdwdGpyYyIsImlhdCI6MTczMzE1ODU0MSwiZXhwIjoxNzQwNzAwODAwLCJpc19yZXZva2VkIjpmYWxzZSwiYWNjb3VudF9pZCI6IjAwMzJlNDQ4LTBlNTctNGQ1Mi05M2NjLTcwMTJhOTI5ZTgzOSIsInVzZXJuYW1lIjoibWFudWVsLmJlbml0ZXotc2FuY2hlekBleHQuZWMuZXVyb3BhLmV1IiwicHJvamVjdF9pZCI6IklJRCIsInF1b3RhcyI6W3sibW9kZWxfbmFtZSI6ImdwdC0zNS10dXJiby0wMzAxIiwiZXhwaXJhdGlvbl9mcmVxdWVuY3kiOiJkYWlseSIsInZhbHVlIjo0MDAwMDB9LHsibW9kZWxfbmFtZSI6ImdwdC0zNS10dXJiby0wNjEzIiwiZXhwaXJhdGlvbl9mcmVxdWVuY3kiOiJkYWlseSIsInZhbHVlIjo0MDAwMDB9LHsibW9kZWxfbmFtZSI6ImdwdC0zNS10dXJiby0xNmsiLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH0seyJtb2RlbF9uYW1lIjoiZ3B0LTQiLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH0seyJtb2RlbF9uYW1lIjoiZ3B0LTQtMzJrIiwiZXhwaXJhdGlvbl9mcmVxdWVuY3kiOiJkYWlseSIsInZhbHVlIjo0MDAwMDB9LHsibW9kZWxfbmFtZSI6ImdwdC0zNS10dXJiby0xMTA2IiwiZXhwaXJhdGlvbl9mcmVxdWVuY3kiOiJkYWlseSIsInZhbHVlIjo0MDAwMDB9LHsibW9kZWxfbmFtZSI6ImdwdC00LTExMDYiLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH1dfQ.F4fDws4IjuvJrbd-v6-uxR9Fzn1F510gf2VlkCyP3m4"
L_token= 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImM5YjY1NDAyLTM3NGItNGZmOS1iZDY2LTZkOGYwYzEzN2ZlZiIsImlzcyI6ImdwdGpyYyIsImlhdCI6MTczNDA5MDA1NSwiZXhwIjoxNzM4MjgxNjAwLCJpc19yZXZva2VkIjpmYWxzZSwiYWNjb3VudF9pZCI6IjFmMTIzMjkzLTc1NjctNDY1Yi04ODhiLTVhYTlmNzEyYzc2YSIsInVzZXJuYW1lIjoibG9yZW56by5uYXBvbGl0YW5vQGVjLmV1cm9wYS5ldSIsInByb2plY3RfaWQiOiJJSUQiLCJkZXBhcnRtZW50IjoiSlJDLlQuNiIsInF1b3RhcyI6W3sibW9kZWxfbmFtZSI6ImdwdC0zNS10dXJiby0wMzAxIiwiZXhwaXJhdGlvbl9mcmVxdWVuY3kiOiJkYWlseSIsInZhbHVlIjo0MDAwMDB9LHsibW9kZWxfbmFtZSI6ImdwdC0zNS10dXJiby0wNjEzIiwiZXhwaXJhdGlvbl9mcmVxdWVuY3kiOiJkYWlseSIsInZhbHVlIjo0MDAwMDB9LHsibW9kZWxfbmFtZSI6ImdwdC0zNS10dXJiby0xNmsiLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH0seyJtb2RlbF9uYW1lIjoiZ3B0LTQiLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH0seyJtb2RlbF9uYW1lIjoiZ3B0LTQtMzJrIiwiZXhwaXJhdGlvbl9mcmVxdWVuY3kiOiJkYWlseSIsInZhbHVlIjo0MDAwMDB9LHsibW9kZWxfbmFtZSI6ImdwdC0zNS10dXJiby0xMTA2IiwiZXhwaXJhdGlvbl9mcmVxdWVuY3kiOiJkYWlseSIsInZhbHVlIjo0MDAwMDB9LHsibW9kZWxfbmFtZSI6ImdwdC00LTExMDYiLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH1dLCJhY2Nlc3NfZ3JvdXBzIjpbeyJhY2Nlc3NfZ3JvdXAiOiJnZW5lcmFsIn1dfQ.BR7NXKBWQhINc3JGX4ShNYJx5HOWM68Jgaer4KCJ18Q'
victor = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjNkOTEzMDgwLTc4NGUtNDIyMi1hZGI5LThlNDQwNjExY2M5MyIsImlzcyI6ImdwdGpyYyIsImlhdCI6MTczNjg0ODg0MiwiZXhwIjoxNzQwOTYwMDAwLCJpc19yZXZva2VkIjpmYWxzZSwiYWNjb3VudF9pZCI6IjMwNjIyY2FmLTdhZjQtNGQ3NS1hNTcxLTExZTg0Yjg5N2JjZiIsInVzZXJuYW1lIjoiVmljdG9yLlBPTkZFUlJBREFAZWMuZXVyb3BhLmV1IiwicHJvamVjdF9pZCI6IklJRCIsImRlcGFydG1lbnQiOiJKUkMuVC42IiwicXVvdGFzIjpbeyJtb2RlbF9uYW1lIjoiZ3B0LTRvIiwiZXhwaXJhdGlvbl9mcmVxdWVuY3kiOiJkYWlseSIsInZhbHVlIjo0MDAwMDB9LHsibW9kZWxfbmFtZSI6ImdwdC0zNS10dXJiby0xMTA2IiwiZXhwaXJhdGlvbl9mcmVxdWVuY3kiOiJkYWlseSIsInZhbHVlIjo0MDAwMDB9LHsibW9kZWxfbmFtZSI6ImdwdC00LTExMDYiLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH1dLCJhY2Nlc3NfZ3JvdXBzIjpbeyJhY2Nlc3NfZ3JvdXAiOiJnZW5lcmFsIn1dfQ.L-fqyAcZlJxpZKoPcWTH0R4LNooa_QuVu0ES1dBs4_k'
BASE_URL = "https://api-gpt.jrc.ec.europa.eu/v1"
MAX_WORKERS = 5
BATCH_SIZE = 100 

FOLDER = 'patents_complx/data_by_country_2'

class TokenManager:
    def __init__(self):
        self.tokens = [
            MY_TOKEN,
            TOKEN,
            victor,
            L_token
        ]
        self.current_token_index = 0
        self.exhausted_tokens = set()
        
    def get_current_token(self):
        return self.tokens[self.current_token_index]
    
    def rotate_token(self):
        """Rotate to next available token"""
        self.current_token_index = (self.current_token_index + 1) % len(self.tokens)
        return self.get_current_token()
    
    def mark_token_exhausted(self, token):
        """Mark a token as exhausted"""
        self.exhausted_tokens.add(token)
    
    def all_tokens_exhausted(self):
        """Check if all tokens are exhausted"""
        return len(self.exhausted_tokens) == len(self.tokens)
    
    def reset_exhausted_tokens(self):
        """Reset the exhausted tokens set"""
        self.exhausted_tokens.clear()

# Create global instance
token_manager = TokenManager()

def wait_until_tomorrow():
    """Wait until midnight of the next day"""
    now = datetime.now()
    tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    wait_seconds = (tomorrow - now).total_seconds()
    print(f"Waiting until tomorrow ({tomorrow}) to reset rate limits...")
    sleep(wait_seconds)
    token_manager.reset_exhausted_tokens()

def create_output_directory(file_path):
    """Create output directory based on parquet file name"""
    file_name = Path(file_path).stem  # Gets filename without extension
    output_dir = Path(FOLDER) / file_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_processed_df(original_df, new_columns):
    """
    Helper function to create DataFrame with only processed records
    Returns DataFrame containing only the processed records with proper length matching
    """
    # Obtener la longitud mínima entre todas las columnas nuevas
    min_length = min(len(values) for values in new_columns.values())
    
    # Crear un DataFrame solo con las filas procesadas usando la longitud mínima
    processed_df = original_df.iloc[:min_length].copy()
    
    # Añadir las columnas procesadas, truncando a la longitud mínima
    for col, values in new_columns.items():
        if col in ['llm_latitude', 'llm_longitude']:
            # Manejo especial para columnas numéricas
            processed_values = []
            for val in values[:min_length]:  # Truncar a la longitud mínima
                try:
                    processed_values.append(float(val) if val is not None else None)
                except (ValueError, TypeError):
                    processed_values.append(None)
            processed_df[col] = processed_values
        else:
            processed_df[col] = values[:min_length]  # Truncar a la longitud mínima
            
    # Eliminar filas donde todas las columnas LLM son None
    llm_columns = list(new_columns.keys())
    processed_df = processed_df.dropna(subset=llm_columns, how='all')
    
    return processed_df

def safe_save_dataframe(df, output_path, prefix=''):
    """
    Helper function to safely save DataFrame with fallback to CSV
    Returns: tuple (success: bool, saved_path: Path)
    """
    parquet_path = output_path / f'{prefix}.parquet'
    csv_path = output_path / f'{prefix}.csv'
    
    try:
        # Intenta guardar como parquet
        df.to_parquet(parquet_path)
        print(f"Successfully saved to parquet: {parquet_path}. Records saved: {len(df)}")
        return True, parquet_path
    except Exception as e:
        print(f"Error saving to parquet: {str(e)}")
        try:
            # Intenta guardar como CSV si falla parquet
            df.to_csv(csv_path, index=False)
            print(f"Fallback: Saved to CSV: {csv_path}. Records saved: {len(df)}")
            return True, csv_path
        except Exception as csv_error:
            print(f"Error saving to CSV: {str(csv_error)}")
            return False, None

def parse_llm_response(response):
    try:
        cleaned = response.encode('utf-8').decode('unicode_escape')
        cleaned = cleaned.replace('\n', '').strip()
        parsed_list = json.loads(cleaned)
        return parsed_list
    except json.JSONDecodeError as e:
        print(f"Error parsing: {str(e)}")
        print(f'Address cleeaned: {str(cleaned)}')
        try:
            cleaned = response.replace('ä', 'a').replace('ß', 'ss')
            cleaned = cleaned.replace('\n', '').strip()
            return json.loads(cleaned)
        except:
            new_error = pd.DataFrame([{
                'timestamp': datetime.now(),
                'error': str(e),
                'raw_data': response
            }])
            error_df = pd.concat([error_df, new_error], ignore_index=True)
            return None

def get_coordinates_from_llm(address: List):
    global client
    while True:
        try:
            client = OpenAI(
                api_key=token_manager.get_current_token(),
                base_url=BASE_URL
            )
            
            response = client.chat.completions.create(
                model="gpt-35-turbo-1106",
                messages=[
                    {"role": "system", "content": """
                        Act as a geolocation service. Given a list of addresses from a specified country, provide the geolocation data for each address formatted as follows:              
                        Requirements:              
                            - Use DOUBLE QUOTES for all JSON property names
                            - Represent latitude and longitude coordinates with 6 decimal places of precision
                            - Ensure each geolocation entry includes both latitude and longitude fields
                            
                        Example of correct format:                 
                        [{"latitude": 50.8482, 
                        "longitude": 2.8772}]
                     Please ensure the output adheres to these specifications.
                    """},
                    {"role": "user", "content": str(address)}
                ],
                temperature=0
            )
            sleep(0.5)
            llm_response = response.choices[0].message.content
            return parse_llm_response(llm_response)
            
        except RateLimitError as e:
            print(e)
            current_token = token_manager.get_current_token()
            print(f"Rate limit exceeded for token ending in ...{current_token[-10:]}")
            token_manager.mark_token_exhausted(current_token)
            
            if token_manager.all_tokens_exhausted():
                print("All tokens exhausted. Waiting until tomorrow to reset rate limits...")
                wait_until_tomorrow()
            else:
                token_manager.rotate_token()
                print(f"Switching to token ending in ...{token_manager.get_current_token()[-10:]}")
                continue
                
        except Exception as e:
            print(f"Error processing address {address}: {str(e)}")
            return []

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

def process_parquet_in_batches(df, output_path, batch_size=10, checkpoint_frequency=500, previous_records=0):
    new_columns = {
        'llm_latitude': [],
        'llm_longitude': []
    }
    
    total_processed = 0
    batch_processed = 0
    
    try:
        for start in range(0, len(df), batch_size):
            end = start + batch_size
            batch = df.iloc[start:end]
            current_batch_size = len(batch)
            
            try:
                coordinates_list = get_coordinates_from_llm(batch['person_address'].tolist())
                
                if coordinates_list:
                    # Verificar que tenemos la cantidad correcta de resultados
                    if len(coordinates_list) == current_batch_size:
                        for coord in coordinates_list:
                            if coord:
                                for col in new_columns:
                                    field = col.replace('llm_', '')
                                    new_columns[col].append(coord.get(field))
                                batch_processed += 1
                            else:
                                # Si no hay coordenadas válidas, añadir None
                                for col in new_columns:
                                    new_columns[col].append(None)
                    else:
                        print(f"Warning: Received {len(coordinates_list)} results for batch of {current_batch_size}")
                        # Ajustar el batch al número de resultados recibidos
                        for coord in coordinates_list:
                            if coord:
                                for col in new_columns:
                                    field = col.replace('llm_', '')
                                    new_columns[col].append(coord.get(field))
                                batch_processed += 1
                
                # Regular checkpoint
                if start % checkpoint_frequency == 0 and new_columns['llm_latitude']:
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
                    
            except RateLimitError:
                if new_columns['llm_latitude']:
                    processed_df = get_processed_df(df.iloc[:start + current_batch_size], new_columns)
                    total_processed = len(processed_df)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    safe_save_dataframe(
                        processed_df,
                        output_path,
                        f'ratelimit_{start + previous_records}_{timestamp}'
                    )
                    print(f"Rate limit reached. Progress saved. Total records: {total_processed + previous_records}")
                return processed_df, "ratelimit"
                
            except Exception as e:
                print(f"Error in batch starting at {start}: {str(e)}")
                continue
        
        if new_columns['llm_latitude']:
            final_df = get_processed_df(df, new_columns)
            total_processed = len(final_df)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_save_dataframe(final_df, output_path, f'final_results_{timestamp}')
            print(f"Process completed. Total records: {total_processed + previous_records}")
            return final_df, "success"
        else:
            print("No records were successfully processed")
            return None, "error"
        
    except Exception as e:
        print(f"Uncaught exception: {str(e)}")
        if new_columns['llm_latitude']:
            processed_df = get_processed_df(df, new_columns)
            return processed_df, "error"
        return None, "error"

def process_parquet_file(file_name):
    file_path = Path(FOLDER) / file_name
    
    try:
        output_dir = create_output_directory(file_path)
        print(f"Created/accessed output directory: {output_dir}")
        
        # Check for existing processed files
        start_index, last_df = get_last_processed_record(output_dir)
        previous_records = len(last_df) if last_df is not None else 0
        
        if start_index >= 0:
            print(f"Found existing processed data. Resuming from index {start_index}")
            print(f"Previously processed records: {previous_records}")
            
            # Read the original dataframe
            df = pd.read_parquet(file_path)
            # Process remaining records
            remaining_df = df.iloc[start_index:]
            print(f"Remaining records to process: {len(remaining_df)}")
            
            processed_df, status = process_parquet_in_batches(
                remaining_df, 
                output_dir,
                previous_records=previous_records
            )
            
            if processed_df is not None and last_df is not None:
                # Combine previous and new results
                final_df = pd.concat([last_df, processed_df], ignore_index=True)
                print(f"Combined previous ({len(last_df)} records) and new ({len(processed_df)} records) results")
            else:
                final_df = processed_df or last_df
        else:
            print("No existing processed data found. Starting from beginning")
            df = pd.read_parquet(file_path)
            final_df, status = process_parquet_in_batches(df, output_dir)
        
        if final_df is not None:
            print(f"Processed: {file_path}")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save based on status
            if status == "success":
                safe_save_dataframe(final_df, output_dir, f'geo_results_{timestamp}')
            elif status == "error":
                safe_save_dataframe(final_df, output_dir, f'error_{timestamp}')
            elif status == "ratelimit":
                safe_save_dataframe(final_df, output_dir, f'ratelimit_final_{timestamp}')
        else:
            print("No records were processed successfully")
        
    except KeyboardInterrupt:
        print("Process interrupted by user.")
        if 'final_df' in locals() and final_df is not None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_save_dataframe(final_df, output_dir, f'interrupt_{timestamp}')
    
    finally:
        print("Finalizing script...")
        if 'final_df' in locals() and final_df is not None:
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

country_files = find_country_parquet_files(FOLDER)
countries_excluded = ['CY.parquet', 'EE.parquet', 'HR.parquet', 'LT.parquet', 'LV.parquet','MT.parquet','SK.parquet', 'PL.parquet', 'GR.parquet', 'ES.parquet', 'SE.parquet', 'RO.parquet', 'AT.parquet','BE.parquet', 'FI.parquet', 'DK.parquet', 'BG.parquet', 'IT.parquet', 'SI.parquet', 'CZ.parquet','HU.parquet']
"""
Fi y Dk hay que arregarlos
"""
for country in country_files:
    if country not in countries_excluded:
        print(f'Procesing {country} file')
        process_parquet_file(country)
        print(f'Proces finished for file {country}')
print("Process completed and saved")