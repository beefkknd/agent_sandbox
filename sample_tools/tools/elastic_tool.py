"""
This module exposes Elasticsearch client methods as MCP tools, with outputs
formatted for consumption by Large Language Models.
"""

import os
import json
from typing import Dict, Any, List
from .elastic import ElasticSearchClientException, ElasticSearchClient

base_url = os.environ.get('ELASTICSEARCH_URL', 'http://localhost:9200')
username = os.environ.get('ELASTICSEARCH_USERNAME', 'elastic')
password = os.environ.get('ELASTICSEARCH_PASSWORD', 'password123')

# Client Initialization
elastic_client = ElasticSearchClient(base_url=base_url, auth=(username, password))

print(f"ElasticSearchClient initialized with URL: {base_url}")
print(f"Elastic indices available: {elastic_client.get_all_indices()}")

# Private Helper Functions
def _format_es_error(e: ElasticSearchClientException) -> str:
    return f"Error: Elasticsearch API call failed. Details: {str(e)}"

def _remove_empty_values(data: Any) -> Any:
    """
    Recursively removes fields with null, 'null', '', {}, or [] values from dictionaries and lists.
    """
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            # Recursively clean the value first
            cleaned_value = _remove_empty_values(value)

            # Skip if value is null, 'null', '', {}, or []
            if cleaned_value is None or cleaned_value == 'null' or cleaned_value == '' or cleaned_value == {} or cleaned_value == []:
                continue

            cleaned[key] = cleaned_value
        return cleaned
    elif isinstance(data, list):
        cleaned_list = []
        for item in data:
            cleaned_item = _remove_empty_values(item)
            # Skip if item is null, 'null', '', {}, or []
            if cleaned_item is None or cleaned_item == 'null' or cleaned_item == '' or cleaned_item == {} or cleaned_item == []:
                continue
            cleaned_list.append(cleaned_item)
        return cleaned_list
    else:
        return data

def _parse_properties(properties: Dict[str, Any], indent_level: int = 0) -> List[str]:
    """Recursively parses mapping properties into indented list of strings."""
    lines = []
    for field_name, field_data in properties.items():
        indent= "  " * indent_level
        if 'properties' in field_data and isinstance(field_data['properties'], dict):
            # Nested properties
            lines.append(f"{indent}{field_name} (object)")
            lines.extend(_parse_properties(field_data['properties'], indent_level + 1))
        else:
            # Regular field
            field_type = field_data.get('type', 'unknown')
            lines.append(f"{indent}{field_name} ({field_type})")
    return lines

def _clean_es_agg_result(agg_data: Dict[str, Any]) -> Any:
    """Recursively cleans ES aggregation results, simplifying structure for better readability."""
    # Simple metric aggregations (e.g., avg, sum, max, min)
    if 'value' in agg_data:
        return agg_data['value']

    # Bucket aggregations (e.g., terms, histogram, date_histogram)
    if 'buckets' in agg_data:
        cleaned_buckets = []
        for bucket in agg_data['buckets']:
            cleaned_bucket = {
                'key': bucket.get('key_as_string', bucket.get('key')),
                'doc_count': bucket.get('doc_count')
            }
            # Process sub-aggregations within the bucket
            for key, value in bucket.items():
                if key not in ('key', 'key_as_string', 'doc_count') and isinstance(value, dict):
                    cleaned_bucket[key] = _clean_es_agg_result(value)
            cleaned_buckets.append(cleaned_bucket)
        return cleaned_buckets

    # Top hits aggregations - extract _source from documents
    if 'hits' in agg_data and 'hits' in agg_data['hits']:
        return [hit.get('_source', {}) for hit in agg_data['hits']['hits']]

    # Recursively process nested dictionaries (fallback)
    cleaned_data = {}
    for key, value in agg_data.items():
        if key == 'meta':  # Skip metadata fields
            continue
        if isinstance(value, dict):
            cleaned_data[key] = _clean_es_agg_result(value)
        else:
            cleaned_data[key] = value

    return cleaned_data if cleaned_data else agg_data

def _format_aggregation_es_json(aggregations: Dict[str, Any]) -> Dict[str, Any]:
    """Formats Elasticsearch aggregation results into a clean, simplified JSON structure."""
    if not aggregations:
        return {}
    return {agg_name: _clean_es_agg_result(agg_data) for agg_name, agg_data in aggregations.items()}

def _format_aggregations(aggregations: Dict[str, Any]) -> List[str]:
    """Formats aggregation results, showing a summary and a sample of buckets."""
    lines = []
    if aggregations:
        for agg_name, agg_data in aggregations.items():
            if 'value' in agg_data and agg_data['value'] is not None:
                lines.append(f"- {agg_name}: {agg_data['value']}")
            elif 'buckets' in agg_data:
                buckets = agg_data['buckets']
                total_buckets = len(buckets)
                lines.append(f"- {agg_name} (Showing min({total_buckets}, 7)) of {total_buckets} buckets):")
                for bucket in buckets[:7]:
                    lines.append(f"  - Key: {bucket['key']}: {bucket['doc_count']}")
                    if bucket.get('aggregations'):
                        lines.extend(["    " + line for line in _format_aggregations(bucket['aggregations'])])
    return lines

# Public MCP Tools
def get_all_elastic_indices() -> str:
    """
    Retrieves all healthy, non-system indices from Elasticsearch and returns
    them as a comma-separated string.
    """
    try:
        print(f"get_all_indices ...")
        # This assumes the cat/indices API returns a list of dicts similar to
        # the JSON format of the cat/indices API.
        indices = elastic_client.get_all_indices()
        healthy_indices = [
            idx['index'] for idx in indices
            if idx.get('health') in ('green', 'yellow') and not idx.get('index', '').startswith('.')
        ]
        if not healthy_indices:
            return "No healthy, non-system indices found."
        return ', '.join(healthy_indices)
    except ElasticSearchClientException as e:
        return _format_es_error(e)

def get_elastic_index_mapping(index_name: str) -> str:
    """
    Retrieves the mapping for a given index and formats it into an indented,
    human-readable structure for LLM.
    """
    try:
        print("get_index_mapping ...")
        mapping = elastic_client.get_index_mapping(index_name=index_name)
        # The actual properties are usually nested under the index name and 'mappings'
        index_mapping = mapping.get(index_name, {}).get('mappings', {})
        properties = index_mapping.get('properties', {})

        if not properties:
            return f"No explicit mapping properties found for index '{index_name}'."
        return '\n'.join(_parse_properties(properties))
    except ElasticSearchClientException as e:
        return _format_es_error(e)

def _prepare_aggs_for_query(aggs: Dict[str, Any]) -> None:
    """Recursively sets a default size of 100 for bucket aggregations if not specified."""
    for agg_body in aggs.values():
        # Set default size for common bucket aggregations
        for agg_type in ['terms', 'significant_terms', 'sample']:
            if agg_type in agg_body and isinstance(agg_body[agg_type], dict):
                agg_body[agg_type].setdefault('size', 100)
                break

        # Recursively process nested aggregations
        nested_aggs = agg_body.get('aggs') or agg_body.get('aggregations')
        if nested_aggs and isinstance(nested_aggs, dict):
            _prepare_aggs_for_query(nested_aggs)

def _sanitize_query(query_dict: Dict[str, Any]) -> None:
    """Enforces a size limit of 10 on document hits to prevent excessive results."""
    current_size = query_dict.get('size')
    is_agg_only = 'aggs' in query_dict or 'aggregations' in query_dict

    if current_size is not None:
        try:
            query_dict['size'] = min(int(current_size), 10)
        except (ValueError, TypeError):
            pass
    elif not is_agg_only:
        query_dict['size'] = 10 
        

def run_elastic_query(index_name: str, query: str | Dict[str, Any], output_format: str = 'json') -> str:
    """
    Runs a query on a given Elasticsearch index and formats the results into
    a human readable summary format for LLM.
    This function applies the following rules:
    - For document hits, limit the size to 10.
    - For aggregations, show the total number of buckets but display only the top 7.
    Args:
        index_name: The name of the index to query.
        query: A JSON string or a dictionary for the Elasticsearch query body.
    Returns:
        A string containing the formatted search results.
    """
    query_dict = None
    if isinstance(query, str):
        try:
            # Parse the incoming query string into a dictionary
            query_dict = json.loads(query)
        except json.JSONDecodeError:
            return "Error: Provided query is not a valid JSON string. Please provide a valid."
    else:
        query_dict = query
    
    is_agg_query = 'aggs' in query_dict or 'aggregations' in query_dict

    # Sanitize the query
    _sanitize_query(query_dict)

    if is_agg_query:
        aggs_part = query_dict.get('aggs') or query_dict.get('aggregations')
        if isinstance(aggs_part, dict):
            _prepare_aggs_for_query(aggs_part)  

    prepared_query_str = json.dumps(query_dict)        
    print(f"======> Original Query {index_name}: {query}")
    print(f"======> Prepared Query {index_name} : {prepared_query_str}")

    try:
        results = elastic_client.run_elastic_query(index_name, prepared_query_str)

        if output_format == 'json':
            if 'aggregations' in results:
                results['aggregations'] = _format_aggregation_es_json(results['aggregations'])
            # Remove null/empty fields before returning
            results = _remove_empty_values(results)
            return json.dumps(results, indent=2)

        # Format the results for text output
        hits = results.get('hits', {})
        total_hits = hits.get('total', {}).get('value', 0)

        output_lines = [f"Total hits: {total_hits}"]
        # Always show total hits. It's useful context for the LLM
        # Format documents if they are present
        if hits.get('hits'):
            for i, doc in enumerate(hits.get('hits', []), start=1):
                source = doc.get('_source', {})
                output_lines.append(f"Record {i}:")
                for key, value in source.items():
                    if value is not None and value != '':
                        output_lines.append(f"   - {key}: {value}")
            output_lines.append("")

        # Format aggregations if they are present
        if 'aggregations' in results:
            output_lines.extend(_format_aggregations(results['aggregations']))
        return '\n'.join(output_lines)
    except ElasticSearchClientException as e:
        print(f"Error executing query in {index_name}:")
        print(f"Query was: {prepared_query_str}")
        print(f"Error details: {e}")
        return _format_es_error(e)
    except Exception as e:
        print(f"Unexpected error executing query in {index_name}: {e}")
        return f"Error: An unexpected error occurred. Details: {str(e)}"