"""Elasticsearch utility"""

from pprint import pprint
from os import environ as env

from elasticsearch import Elasticsearch, NotFoundError, RequestError

from paper_chat.core.configs import CONFIGS_ES


class ElasticSearchManager:
    def __init__(self, reset: bool = False, mappings: dict = CONFIGS_ES.mappings):
        self.es = Elasticsearch(
            hosts=CONFIGS_ES.connection.hosts,
            basic_auth=(CONFIGS_ES.connection.username, env["ELASTIC_PASSWORD"]),
            verify_certs=False,
        )
        print("Connected to Elasticsearch!")

        if reset:
            self.reset_index()
        if mappings:
            self.create_index(mappings)

        # client_info = self.es.info()
        # pprint(client_info.body)

    def create_index(self, mappings: dict):
        for index, mapping in mappings.items():
            if not self.es.indices.exists(index=index):
                self.es.indices.create(index=index, body={"mappings": mapping})
                print(f"Index {index} created successfully.")
            else:
                print(f"Index {index} already exists. No changes were made.")

    def reset_index(self):
        try:
            # Get all indices
            indices = self.es.indices.get(index="*")

            # Delete each index
            for index in indices:
                print(f"Deleting index: {index}")
                self.es.indices.delete(index=index, ignore=[400, 404])

            print("All indices have been deleted.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def insert_document(self, document, index: str):
        return self.es.index(index=index, body=document)

    def insert_documents(self, documents, index: str):
        # NOTE: see details in https://elasticsearch-py.readthedocs.io/en/stable/api.html#elasticsearch.Elasticsearch.bulk
        operations = []
        for document in documents:
            # For each document, two entries are added to the operations list:
            #   1. A description of what operation to perform, set to index, with the name of the index given as an argument.
            #   2. The actual data of the document
            operations.append({"index": {"_index": index}})
            operations.append(document)
        return self.es.bulk(operations=operations)

    def search(self, index: str, **query_args):
        try:
            response = self.es.search(index=index, **query_args)
            # print(response["hits"]["total"])
            # for hit in response["hits"]["hits"]:
            #     print(hit["_source"])
            return response
        except NotFoundError:
            print(f"Index({index}) does not exist.")
            return

    def search_id(self, index: str, **query_args) -> str:
        response = self.search(index, **query_args)
        if response:
            hits = response["hits"]["hits"]
            if hits and len(hits) == 1:
                return hits[0]["_id"]
        return ""

    def search_hit(self, index: str, **query_args):
        # TODO
        raise NotImplementedError

    def update(self, index: str, id: str, document: dict):
        try:
            response = self.es.update(
                index=index, id=id, body={"doc": document, "doc_as_upsert": True}
            )
            print(f"Document {id} updated successfully")
            return response
        except NotFoundError:
            print(f"Document with id {id} not found in index {index}")
            return None
        except RequestError as e:
            print(f"Error updating document: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error occurred: {str(e)}")
            return None

    def retrieve_document(self, index: str, id: str):
        return self.es.get(index=index, id=id)


if __name__ == "__main__":
    es_manager = ElasticSearchManager(reset=True)
