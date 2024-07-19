"""Elasticsearch utility"""

from pprint import pprint
from os import environ as env

from elasticsearch import Elasticsearch, NotFoundError, RequestError
from elasticsearch.helpers import bulk

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paper_chat.core.utils import MetaSingleton
from paper_chat.core.llm import EMBEDDINGS


class ElasticSearchManager(metaclass=MetaSingleton):
    def __init__(self, reset: bool = False):
        self.es = Elasticsearch(
            hosts="https://es01:9200",
            basic_auth=("elastic", env["ELASTIC_PASSWORD"]),
            verify_certs=False,
        )
        client_info = self.es.info()
        if reset:
            self.reset_index()

        print("Connected to Elasticsearch!")
        # pprint(client_info.body)

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

    def search_id(self, index: str, **query_args):
        response = self.search(index, **query_args)
        if response:
            ids = [doc["_id"] for doc in response["hits"]["hits"]]
        else:
            ids = [""]

        if len(ids) > 1:
            raise ValueError(f"Multiple documents found: {ids}")
        return ids[0]

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
    es_manager = ElasticSearchManager()
    # arxiv_url = "https://arxiv.org/pdf/1706.03762"
    # es_manager.insert_paper(arxiv_url)

    # id = es_manager.search_id(
    #     INDEX_PAPERS_METADATA, body={"query": {"match": {"title": "Attention"}}}
    # )
    # print(id)
    # es_manager.update(index=INDEX_PAPERS_METADATA, id=id, document={"summary": "abcd"})
