
import csv
import zipfile
import io
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Elasticsearch connection settings
ES_HOST = "localhost"
ES_PORT = 9200
ES_INDEX = "ais_index"

# Path to the data file
ZIP_FILE_PATH = "AIS_2022_01_01.zip"

# Elasticsearch mapping
ES_MAPPING = {
    "properties": {
        "BaseDateTime": {"type": "date"},
        "COG": {"type": "float"},
        "CallSign": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
        "Cargo": {"type": "long"},
        "Draft": {"type": "float"},
        "Heading": {"type": "float"},
        "IMO": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
        "LAT": {"type": "float"},
        "LON": {"type": "float"},
        "Length": {"type": "long"},
        "MMSI": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
        "SOG": {"type": "float"},
        "Status": {"type": "long"},
        "TransceiverClass": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
        "VesselName": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
        "VesselType": {"type": "long"},
        "Width": {"type": "long"},
        "location": {"type": "geo_point"}
    }
}

def generate_actions(zip_file_path, doc_limit=None):
    """
    Generator function to yield documents for bulk indexing.
    Reads a CSV from a zip file in memory, processes, and yields each row.
    Optionally limits the number of documents yielded.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zf:
        csv_file_name = next((name for name in zf.namelist() if name.lower().endswith('.csv')), None)

        if not csv_file_name:
            print("No CSV file found in the zip archive.")
            return

        print(f"Found CSV file: {csv_file_name}")

        with zf.open(csv_file_name, 'r') as csv_file:
            reader = csv.DictReader(io.TextIOWrapper(csv_file, 'utf-8'))
            doc_count = 0
            for row in reader:
                if doc_limit is not None and doc_count >= doc_limit:
                    break
                try:
                    lat = row.get('LAT')
                    lon = row.get('LON')

                    if not lat or not lon:
                        print(f"Skipping row due to missing LAT or LON. Row: {row}")
                        continue

                    row["location"] = f"{lat},{lon}"
                    
                    doc = {
                        "_index": ES_INDEX,
                        "_source": row
                    }
                    
                    for key, value in doc["_source"].items():
                        if key in ES_MAPPING["properties"] and isinstance(value, str):
                            field_type = ES_MAPPING["properties"][key]["type"]
                            if value.strip() == "":
                                doc["_source"][key] = None
                                continue
                            try:
                                if field_type in ["float", "double", "half_float", "scaled_float"]:
                                    doc["_source"][key] = float(value)
                                elif field_type in ["long", "integer", "short", "byte"]:
                                    doc["_source"][key] = int(float(value))
                            except (ValueError, TypeError):
                                pass
                    
                    yield doc
                    doc_count += 1

                except (ValueError, KeyError) as e:
                    print(f"Skipping row due to error: {e}. Row: {row}")
                    continue

def main():
    """
    Main function to connect to Elasticsearch, create index with mapping, and run bulk indexing.
    """
    try:
        es_client = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT, 'scheme': 'http'}])

        if not es_client.ping():
            print("Could not connect to Elasticsearch. Please ensure it is running.")
            return

        print("Connected to Elasticsearch.")

        if not es_client.indices.exists(index=ES_INDEX):
            print(f"Creating index '{ES_INDEX}' with mapping and settings.")
            
            settings = {
                "index": {
                    "query": {
                        "parse": {
                            "allow_unmapped_fields": False
                        }
                    }
                }
            }
            
            es_client.indices.create(index=ES_INDEX, mappings=ES_MAPPING, settings=settings)
        else:
            print(f"Index '{ES_INDEX}' already exists.")

        print(f"Starting bulk indexing of {ZIP_FILE_PATH} into index '{ES_INDEX}'...")
        
        # Use the bulk helper to index the data, with a limit of 100 for testing
        success, failed = bulk(es_client, generate_actions(ZIP_FILE_PATH))

        print(f"Bulk indexing complete.")
        print(f"Successfully indexed: {success} documents.")
        print(f"Failed to index: {failed} documents.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
