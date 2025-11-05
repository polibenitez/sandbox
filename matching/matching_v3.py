"""
matching_v3.py - Versión mejorada del matching Orbis-Patstat

MEJORAS EN v3:
- Limpieza mejorada de códigos NUTS3 en Orbis (extrae solo primeros 5 caracteres)
- Mejor logging y estadísticas por nivel de matching
- Optimización de memoria con escritura directa a disco
- Soporte para nombres en el output
"""

import polars as pl
import sqlalchemy
import os
import gc
import duckdb
from datetime import datetime

# --- CONFIGURACIÓN ---
ORACLE_CONFIG = {
    "user": "USER_BENMANU",
    "password": "HjKH6GvY3a_yc7Jz",
    "host": "siprodb01p-dars.jrc.it",
    "port": "1525",
    "service_name": "SCIDB",
    'schema_name': 'PROJECT_COMPLEXITY',
}

# --- PARÁMETROS DEL PROCESO ---
EUROPEAN_COUNTRIES_all = [
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR',
    'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK',
    'SI', 'ES', 'SE'
]

# --- PARÁMETROS ---
EUROPEAN_COUNTRIES = ['ES']  # Puedes añadir más países
PATSTAT_PARQUET_PATH = "v_paises/ES.parquet"
OUTPUT_DIR = "v3/resultados_matching"
ORBIS_CACHE_DIR = "scoreboard_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ORBIS_CACHE_DIR, exist_ok=True)

# --- JERARQUÍA DE MATCHING ---
# El orden es importante: de más granular a menos granular.
MATCHING_HIERARCHY = ['postcode', 'nuts3']


def log_with_timestamp(message):
    """Añade timestamp a los mensajes de log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def process_country_matching_v3(country_code: str, connection_str: str):
    """
    Procesamiento optimizado usando DuckDB con mejoras en limpieza de NUTS3.

    Mejoras en v3:
    - NUTS3 en Orbis: extrae solo los primeros 5 caracteres (ej: 'ES432 - Caceres' -> 'ES432')
    - Mejor logging con timestamps
    - Estadísticas detalladas por nivel
    """
    log_with_timestamp(f"Iniciando matching v3 para: {country_code}")

    # Crear conexión DuckDB
    conn = duckdb.connect()
    conn.execute("SET memory_limit='20GB'")
    conn.execute("SET threads TO 16")

    try:
        log_with_timestamp("PASO 1/5 - Cargando datos en DuckDB")

        # Columnas de Orbis (minúsculas como vienen del caché)
        orbis_cols = {
            "id": "bvd_id",
            "name": "name_native",
            "postcode": "postcode",
            "nuts3": "nuts3",
            "country": "ctryiso"
        }

        # Columnas de Patstat (MAYÚSCULAS)
        patstat_cols = {
            "id": "PERSON_ID",
            "name": "PERSON_NAME",
            "postcode": "PERSON_POSTCODE",
            "nuts3": "NUTS3",
            "country": "PERSON_CTRY_CODE"
        }

        # Cargar Orbis desde caché o Oracle
        cache_file_path = os.path.join(ORBIS_CACHE_DIR, f"orbis_{country_code}.parquet")
        if os.path.exists(cache_file_path):
            conn.execute(f"""
                CREATE TABLE orbis AS
                SELECT * FROM read_parquet('{cache_file_path}')
            """)
            log_with_timestamp(f"  ✓ Orbis cargado desde caché: {cache_file_path}")
        else:
            log_with_timestamp("  → Cargando Orbis desde Oracle (puede tardar)...")
            orbis_db_names = {k: v.upper() for k, v in orbis_cols.items()}
            sql_selects = [f'{orbis_db_names[k]} AS "{v}"' for k, v in orbis_cols.items()]

            # Query para Scoreboard
            sql_query = f"""
                SELECT {', '.join([f'ci.{col}' for col in sql_selects])}
                FROM OWN_ORBIS_2024A.CONTACT_INFO ci
                JOIN user_napollo.SB23_FIRMS_TO_MATCH ftm
                ON ci.BVD_ID = ftm.BVD_ID
                WHERE ci.{orbis_db_names['country']} = '{country_code}'
            """

            orbis_df = pl.read_database_uri(query=sql_query, uri=connection_str)

            if not orbis_df.is_empty():
                orbis_df.write_parquet(cache_file_path)
                conn.execute(f"""
                    CREATE TABLE orbis AS
                    SELECT * FROM read_parquet('{cache_file_path}')
                """)
                log_with_timestamp(f"  ✓ Orbis guardado en caché: {orbis_df.height:,} registros")
            else:
                log_with_timestamp("  ✗ No se encontraron datos de Orbis")
                return

        # Cargar Patstat
        log_with_timestamp(f"  → Cargando Patstat: {PATSTAT_PARQUET_PATH}")
        conn.execute(f"""
            CREATE TABLE patstat AS
            SELECT * FROM read_parquet('{PATSTAT_PARQUET_PATH}')
            WHERE PERSON_CTRY_CODE = '{country_code}'
        """)

        # Verificar datos cargados
        patstat_count = conn.execute("SELECT COUNT(*) FROM patstat").fetchone()[0]
        orbis_count = conn.execute("SELECT COUNT(*) FROM orbis").fetchone()[0]
        log_with_timestamp(f"  ✓ Datos cargados - Orbis: {orbis_count:,} | Patstat: {patstat_count:,}")

        if patstat_count == 0:
            log_with_timestamp(f"  ✗ WARNING: No hay datos de Patstat para {country_code}")
            return

        # Detectar tipo de PERSON_ID
        person_id_type = conn.execute("SELECT typeof(PERSON_ID) FROM patstat LIMIT 1").fetchone()[0]
        log_with_timestamp(f"  ℹ Tipo PERSON_ID: {person_id_type}")

        # Crear tabla de IDs matcheados
        conn.execute(f"""
            CREATE TABLE matched_ids (
                person_id {person_id_type},
                match_level VARCHAR
            )
        """)

        # --- PROCESAMIENTO EN CASCADA ---
        log_with_timestamp("PASO 2/5 - Ejecutando matching jerárquico")
        partial_result_files = []
        matching_stats = {}

        for level in MATCHING_HIERARCHY:
            log_with_timestamp(f"\n  → Procesando nivel: '{level.upper()}'")

            # === MEJORA v3: Limpieza mejorada de NUTS3 ===
            if level == 'nuts3':
                # Orbis: Extraer solo los primeros 5 caracteres alfanuméricos
                # Ejemplo: 'ES432 - Caceres' -> 'ES432'
                orbis_std = f"UPPER(REGEXP_EXTRACT({orbis_cols[level]}, '([A-Z0-9]{{5}})', 1))"
                # Patstat: Ya viene limpio, solo uppercase
                patstat_std = f"TRIM(UPPER({patstat_cols[level]}))"
                log_with_timestamp(f"    ℹ NUTS3 - Limpieza mejorada v3 aplicada")
            else:
                # Postcode: limpiar caracteres especiales
                orbis_std = f"UPPER(REGEXP_REPLACE(TRIM({orbis_cols[level]}), '[^A-Z0-9]', '', 'g'))"
                patstat_std = f"UPPER(REGEXP_REPLACE(TRIM({patstat_cols[level]}), '[^A-Z0-9]', '', 'g'))"

            # Archivo temporal para este nivel
            output_path_level = os.path.join(OUTPUT_DIR, f"temp_matched_{country_code}_{level}.parquet")

            # Query de matching con COPY para escritura directa
            query = f"""
                COPY (
                    WITH orbis_clean AS (
                        SELECT
                            "{orbis_cols['id']}" as orbis_id,
                            "{orbis_cols['name']}" as orbis_name,
                            "{orbis_cols['country']}" as orbis_country,
                            "{orbis_cols['postcode']}" as orbis_postcode,
                            "{orbis_cols['nuts3']}" as orbis_nuts3,
                            {orbis_std} as matching_key
                        FROM orbis
                        WHERE {orbis_std} IS NOT NULL
                        AND LENGTH({orbis_std}) > 0
                    ),
                    patstat_clean AS (
                        SELECT
                            {patstat_cols['id']} as person_id,
                            {patstat_cols['name']} as person_name,
                            {patstat_cols['country']} as patstat_country,
                            {patstat_cols['postcode']} as person_postcode,
                            {patstat_cols['nuts3']} as person_nuts3,
                            {patstat_std} as matching_key
                        FROM patstat
                        WHERE {patstat_cols['id']} NOT IN (
                            SELECT person_id FROM matched_ids
                        )
                        AND {patstat_std} IS NOT NULL
                        AND LENGTH({patstat_std}) > 0
                    ),
                    matched AS (
                        SELECT
                            o.orbis_id,
                            o.orbis_name,
                            o.orbis_country,
                            o.orbis_postcode,
                            o.orbis_nuts3,
                            p.person_id,
                            p.person_name,
                            p.patstat_country,
                            p.person_postcode,
                            p.person_nuts3,
                            o.matching_key,
                            '{level}' as match_level
                        FROM orbis_clean o
                        INNER JOIN patstat_clean p
                            ON o.orbis_country = p.patstat_country
                            AND o.matching_key = p.matching_key
                    )
                    SELECT * FROM matched
                ) TO '{output_path_level}' (FORMAT PARQUET, CODEC 'ZSTD');
            """

            try:
                conn.execute(query)

                # Verificar si se creó el archivo y tiene contenido
                if os.path.exists(output_path_level) and os.path.getsize(output_path_level) > 0:
                    # Leer solo IDs para actualizar matched_ids
                    new_ids_df = pl.read_parquet(output_path_level, columns=['person_id'])
                    count = len(new_ids_df)

                    if count > 0:
                        log_with_timestamp(f"    ✓ {count:,} coincidencias encontradas")
                        partial_result_files.append(output_path_level)
                        matching_stats[level] = count

                        # Actualizar tabla de IDs matcheados
                        conn.register('temp_new_ids_view', new_ids_df)
                        conn.execute(f"""
                            INSERT INTO matched_ids
                            SELECT DISTINCT person_id, '{level}'
                            FROM temp_new_ids_view
                        """)
                        conn.unregister('temp_new_ids_view')
                    else:
                        log_with_timestamp(f"    ✗ No se encontraron coincidencias")
                        matching_stats[level] = 0
                        if os.path.exists(output_path_level):
                            os.remove(output_path_level)
                else:
                    log_with_timestamp(f"    ✗ No se encontraron coincidencias")
                    matching_stats[level] = 0
                    if os.path.exists(output_path_level):
                        os.remove(output_path_level)

            except Exception as e:
                log_with_timestamp(f"    ✗ ERROR en nivel {level}: {e}")
                matching_stats[level] = 0
                continue

            gc.collect()

        # --- CONSOLIDAR RESULTADOS ---
        if partial_result_files:
            log_with_timestamp("PASO 3/5 - Consolidando resultados matcheados")

            # Combinar todos los archivos parciales
            combined_lazy_df = pl.scan_parquet(partial_result_files)
            output_path = os.path.join(OUTPUT_DIR, f"matched_{country_code}.parquet")
            combined_lazy_df.sink_parquet(output_path)

            total_matches = sum(matching_stats.values())
            log_with_timestamp(f"  ✓ Total matches: {total_matches:,}")
            log_with_timestamp(f"  ✓ Archivo guardado: {output_path}")

            # Mostrar estadísticas por nivel
            log_with_timestamp("  📊 Estadísticas por nivel:")
            for level, count in matching_stats.items():
                percentage = (count / patstat_count * 100) if patstat_count > 0 else 0
                log_with_timestamp(f"    • {level}: {count:,} ({percentage:.2f}%)")

            # Limpiar archivos temporales
            for f in partial_result_files:
                if os.path.exists(f):
                    os.remove(f)
        else:
            log_with_timestamp("  ✗ No se encontraron coincidencias en ningún nivel")

        # --- GUARDAR NO MATCHEADOS ---
        log_with_timestamp("PASO 4/5 - Guardando patentes sin match")
        unmatched_output_path = os.path.join(OUTPUT_DIR, f"unmatched_{country_code}.parquet")
        conn.execute(f"""
            COPY (
                SELECT * FROM patstat
                WHERE {patstat_cols['id']} NOT IN (SELECT person_id FROM matched_ids)
            ) TO '{unmatched_output_path}' (FORMAT PARQUET, CODEC 'ZSTD');
        """)

        unmatched_count = patstat_count - sum(matching_stats.values())
        log_with_timestamp(f"  ✓ {unmatched_count:,} patentes sin match guardadas")
        log_with_timestamp(f"  ✓ Archivo: {unmatched_output_path}")

        # --- RESUMEN FINAL ---
        log_with_timestamp("PASO 5/5 - Resumen final")
        match_rate = (sum(matching_stats.values()) / patstat_count * 100) if patstat_count > 0 else 0
        log_with_timestamp(f"  📊 Tasa de matching: {match_rate:.2f}%")
        log_with_timestamp(f"  ✓ Proceso completado exitosamente")

    except Exception as e:
        log_with_timestamp(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()

    finally:
        conn.close()


def main():
    """Función principal"""
    log_with_timestamp("="*80)
    log_with_timestamp("MATCHING ORBIS-PATSTAT v3")
    log_with_timestamp("="*80)

    connection_str = f"oracle://{ORACLE_CONFIG['user']}:{ORACLE_CONFIG['password']}@{ORACLE_CONFIG['host']}:{ORACLE_CONFIG['port']}/{ORACLE_CONFIG['service_name']}"

    for country in EUROPEAN_COUNTRIES:
        log_with_timestamp(f"\n{'='*80}")
        log_with_timestamp(f"Procesando país: {country}")
        log_with_timestamp(f"{'='*80}")
        process_country_matching_v3(country, connection_str)

    log_with_timestamp("\n" + "="*80)
    log_with_timestamp("PROCESO COMPLETO")
    log_with_timestamp("="*80)


if __name__ == "__main__":
    main()
