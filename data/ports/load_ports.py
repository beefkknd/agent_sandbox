import csv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Elasticsearch connection settings
ES_HOST = "localhost"
ES_PORT = 9200
ES_INDEX = "port_list"
CSV_FILE_PATH = "/Users/yingzhou/work/agent_sanbox/data/ports/UpdatedPub150.csv"

COLUMN_MAPPING = {
    'port_index_num': 'World Port Index Number',
    'region_name': 'Region Name',
    'port_name': 'Main Port Name',
    'alt_port_name': 'Alternate Port Name',
    'un_locode': 'UN/LOCODE',
    'country_code': 'Country Code',
    'water_body': 'World Water Body',
    'sailing_direction_pub': 'Sailing Direction or Publication',
    'pub_link': 'Publication Link',
    'std_nautical_chart': 'Standard Nautical Chart',
    's57_enc': 'IHO S-57 Electronic Navigational Chart',
    's101_enc': 'IHO S-101 Electronic Navigational Chart',
    'digital_nautical_chart': 'Digital Nautical Chart',
    'tidal_range_m': 'Tidal Range (m)',
    'entrance_width_m': 'Entrance Width (m)',
    'channel_depth_m': 'Channel Depth (m)',
    'anchorage_depth_m': 'Anchorage Depth (m)',
    'cargo_pier_depth_m': 'Cargo Pier Depth (m)',
    'oil_terminal_depth_m': 'Oil Terminal Depth (m)',
    'lng_terminal_depth_m': 'Liquified Natural Gas Terminal Depth (m)',
    'max_vessel_length_m': 'Maximum Vessel Length (m)',
    'max_vessel_beam_m': 'Maximum Vessel Beam (m)',
    'max_vessel_draft_m': 'Maximum Vessel Draft (m)',
    'offshore_max_vessel_length_m': 'Offshore Maximum Vessel Length (m)',
    'offshore_max_vessel_beam_m': 'Offshore Maximum Vessel Beam (m)',
    'offshore_max_vessel_draft_m': 'Offshore Maximum Vessel Draft (m)',
    'harbor_size': 'Harbor Size',
    'harbor_type': 'Harbor Type',
    'harbor_use': 'Harbor Use',
    'shelter': 'Shelter Afforded',
    'restrict_tide': 'Entrance Restriction - Tide',
    'restrict_swell': 'Entrance Restriction - Heavy Swell',
    'restrict_ice': 'Entrance Restriction - Ice',
    'restrict_other': 'Entrance Restriction - Other',
    'overhead_limits': 'Overhead Limits',
    'ukc_management': 'Underkeel Clearance Management System',
    'good_holding_ground': 'Good Holding Ground',
    'turning_area': 'Turning Area',
    'port_security': 'Port Security',
    'eta_message': 'Estimated Time of Arrival Message',
    'quarantine_pratique': 'Quarantine - Pratique',
    'quarantine_sanitation': 'Quarantine - Sanitation',
    'quarantine_other': 'Quarantine - Other',
    'traffic_sep_scheme': 'Traffic Separation Scheme',
    'vts': 'Vessel Traffic Service',
    'first_port_of_entry': 'First Port of Entry',
    'us_rep': 'US Representative',
    'pilot_compulsory': 'Pilotage - Compulsory',
    'pilot_available': 'Pilotage - Available',
    'pilot_local_assist': 'Pilotage - Local Assistance',
    'pilot_advisable': 'Pilotage - Advisable',
    'tugs_salvage': 'Tugs - Salvage',
    'tugs_assist': 'Tugs - Assistance',
    'comms_phone': 'Communications - Telephone',
    'comms_fax': 'Communications - Telefax',
    'comms_radio': 'Communications - Radio',
    'comms_radiotelephone': 'Communications - Radiotelephone',
    'comms_airport': 'Communications - Airport',
    'comms_rail': 'Communications - Rail',
    'search_rescue': 'Search and Rescue',
    'navarea': 'NAVAREA',
    'fac_wharves': 'Facilities - Wharves',
    'fac_anchorage': 'Facilities - Anchorage',
    'fac_dangerous_cargo_anchorage': 'Facilities - Dangerous Cargo Anchorage',
    'fac_med_mooring': 'Facilities - Med Mooring',
    'fac_beach_mooring': 'Facilities - Beach Mooring',
    'fac_ice_mooring': 'Facilities - Ice Mooring',
    'fac_roro': 'Facilities - Ro-Ro',
    'fac_solid_bulk': 'Facilities - Solid Bulk',
    'fac_liquid_bulk': 'Facilities - Liquid Bulk',
    'fac_container': 'Facilities - Container',
    'fac_breakbulk': 'Facilities - Breakbulk',
    'fac_oil_terminal': 'Facilities - Oil Terminal',
    'fac_lng_terminal': 'Facilities - LNG Terminal',
    'fac_other': 'Facilities - Other',
    'med_facilities': 'Medical Facilities',
    'garbage_disposal': 'Garbage Disposal',
    'chem_hold_tank_disposal': 'Chemical Holding Tank Disposal',
    'degaussing': 'Degaussing',
    'dirty_ballast_disposal': 'Dirty Ballast Disposal',
    'cranes_fixed': 'Cranes - Fixed',
    'cranes_mobile': 'Cranes - Mobile',
    'cranes_floating': 'Cranes - Floating',
    'cranes_container': 'Cranes - Container',
    'lifts_100_plus_tons': 'Lifts - 100+ Tons',
    'lifts_50_100_tons': 'Lifts - 50-100 Tons',
    'lifts_25_49_tons': 'Lifts - 25-49 Tons',
    'lifts_0_24_tons': 'Lifts - 0-24 Tons',
    'svc_longshoremen': 'Services - Longshoremen',
    'svc_electricity': 'Services - Electricity',
    'svc_steam': 'Services - Steam',
    'svc_nav_equip': 'Services - Navigation Equipment',
    'svc_elec_repair': 'Services - Electrical Repair',
    'svc_ice_breaking': 'Services - Ice Breaking',
    'svc_diving': 'Services - Diving',
    'supp_provisions': 'Supplies - Provisions',
    'supp_water': 'Supplies - Potable Water',
    'supp_fuel_oil': 'Supplies - Fuel Oil',
    'supp_diesel_oil': 'Supplies - Diesel Oil',
    'supp_aviation_fuel': 'Supplies - Aviation Fuel',
    'supp_deck': 'Supplies - Deck',
    'supp_engine': 'Supplies - Engine',
    'repairs': 'Repairs',
    'dry_dock': 'Dry Dock',
    'railway': 'Railway',
    'latitude': 'Latitude',
    'longitude': 'Longitude',
}

def get_es_mapping():
    """
    Generates an Elasticsearch mapping using the short names and stores the original names in the meta field.
    """
    properties = {}
    for short_name, original_name in COLUMN_MAPPING.items():
        field_meta = {"meta": {"source": original_name}}
        if short_name in ['latitude', 'longitude']:
            properties[short_name] = {'type': 'float', **field_meta}
        else:
            properties[short_name] = {'type': 'text', 'fields': {'keyword': {'type': 'keyword', 'ignore_above': 256}}, **field_meta}
    
    properties['port_location'] = {'type': 'geo_point'}

    return {
        "dynamic": "strict",
        "properties": properties
    }

def generate_actions(csv_file_path):
    """
    Generator function to yield documents for bulk indexing using short field names.
    """
    # Create a reverse mapping from original name to short name for faster lookups
    reverse_mapping = {v: k for k, v in COLUMN_MAPPING.items()}

    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_source = {}
            for original_name, value in row.items():
                short_name = reverse_mapping.get(original_name)
                if short_name:
                    # Convert numeric fields
                    if isinstance(value, str) and value.isnumeric():
                        doc_source[short_name] = float(value)
                    else:
                        doc_source[short_name] = value

            try:
                lat = doc_source.get('latitude')
                lon = doc_source.get('longitude')

                if not lat or not lon:
                    continue

                doc_source['port_location'] = {
                    "lat": float(lat),
                    "lon": float(lon)
                }
                
                yield {
                    "_index": ES_INDEX,
                    "_source": doc_source
                }
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

        es_mapping = get_es_mapping()

        if es_client.indices.exists(index=ES_INDEX):
            print(f"Deleting existing index '{ES_INDEX}'.")
            es_client.indices.delete(index=ES_INDEX)

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
        
        es_client.indices.create(index=ES_INDEX, mappings=es_mapping, settings=settings)

        print(f"Starting bulk indexing of {CSV_FILE_PATH} into index '{ES_INDEX}'...")
        
        success, failed = bulk(es_client, generate_actions(CSV_FILE_PATH))

        print("Bulk indexing complete.")
        print(f"Successfully indexed: {success} documents.")
        print(f"Failed to index: {failed} documents.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()