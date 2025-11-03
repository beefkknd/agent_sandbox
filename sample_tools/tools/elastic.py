import requests
from typing import Any, Dict, Tuple
import json

class ElasticSearchClientException(Exception):
    """Custom exception for Elasticsearch client errors."""
    pass

class ElasticSearchClient:
    """A client for interacting with an Elasticsearch server."""

    def __init__(self, base_url: str, auth: Tuple[str, str]):
        """
        Initializes the ElasticSearchClient.

        Args:
            base_url: The base URL of the Elasticsearch server (e.g., "http://localhost:9200").
            auth: A tuple containing the username and password for authentication.
        """
        self.base_url = base_url
        self.auth = auth
        self.headers = {"Content-Type": "application/json"}

    def get_index_mapping(self, index_name: str) -> Dict[str, Any]:
        """
        Retrieves the mapping for a given index from Elasticsearch.

        Args:
            index_name: The name of the index.

        Returns:
            A dictionary containing the index mapping.
        """
        try:
            print(f" aaaaaaa ==========> Getting index mapping for {index_name} ...")
            url = f"{self.base_url}/{index_name}/_mapping"
            response = requests.get(url, auth=self.auth)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_message = f"Failed to get index mapping for index '{index_name}': {e}"
            raise ElasticSearchClientException(error_message) from e

    def get_all_indices(self) -> Dict[str, Any]:
        """
        Retrieves all indices from Elasticsearch.

        Returns:
            A dictionary containing all indices.
        """
        try:
            print(f"aaaaaaa ==========> Getting all indices from {self.base_url} ...")
            url = f"{self.base_url}/_cat/indices?format=json"
            response = requests.get(url, auth=self.auth)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_message = "Failed to get all indices."
            raise ElasticSearchClientException(error_message) from e

    def run_elastic_query(self, index_name: str, query: str) -> Dict[str, Any]:
        """
        Runs a query on a given Elasticsearch index.

        Args:
            index_name: The name of the index.
            query: The Elasticsearch query string.

        Returns:
            A dictionary containing the query result.
        """
        try:
            print(f"aaaaaaa ==========> Running query on index {index_name} {query}...")
            url = f"{self.base_url}/{index_name}/_search"
            response = requests.post(url, auth=self.auth, headers=self.headers, data=query)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_message = f"Failed to run elastic query on index '{index_name}': {e}"
            if e.response is not None:
                try:
                    error_details = e.response.json()
                    error_message += f" | Details: {error_details}"
                except ValueError:
                    error_message += f" | Response: {e.response.text}"
            raise ElasticSearchClientException(error_message) from e
        except json.JSONDecodeError as e:
            raise ElasticSearchClientException(f"Invalid JSON query: {e}") from e