"""Utility"""

import os
import re
from urllib import request
from collections import OrderedDict

import requests
from retry import retry

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import GrobidParser
from langchain_core.document_loaders.base import Document


@retry(delay=10, tries=3)
def fetch_semantic_scholar_data(arxiv_id: str, fields_list: list[str]) -> dict:
    fields = ",".join(fields_list)
    semantic_scholar_url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields={fields}"
    response = requests.get(semantic_scholar_url)
    response.raise_for_status()
    data = response.json()

    result = {}
    for field in fields_list:
        if "." in field:
            key, sub_key = field.split(".")
            if key not in result:
                result[key] = {}
            nested_data = data.get(key)
            if isinstance(nested_data, list):
                item = nested_data[0]
                if isinstance(item[sub_key], list):  # authors.affiliations
                    result[key][sub_key] = [item[sub_key] for item in nested_data]
                elif isinstance(item[sub_key], str):  # authors.name
                    result[key][sub_key] = [item[sub_key] for item in nested_data]
                else:
                    raise NotImplementedError(
                        f"Unsupported data type: {type(item[sub_key])}"
                    )
            elif isinstance(nested_data, dict):
                result[field] = nested_data.get(sub_key)
            else:
                result[field] = None
        else:
            result[field] = data[field]

    return result


def fetch_paper_info_from_url(arxiv_url: str = None, arxiv_id: str = None) -> dict:
    if arxiv_url and not arxiv_id:
        # Extract arXiv ID from URL
        arxiv_id_match = re.search(r"arxiv\.org/pdf/(\d+\.\d+)", arxiv_url)
        if not arxiv_id_match:
            print("Invalid arXiv URL")
            return None
        arxiv_id = arxiv_id_match.group(1)
    elif not arxiv_url and arxiv_id:
        arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}"
    else:
        raise ValueError("Either arxiv_url or arxiv_id should be provided.")

    # Extract data
    paper_info = {"arxiv_id": arxiv_id, "arxiv_url": arxiv_url}
    fields = [
        "title",
        "authors.name",
        "authors.affiliations",
        "venue",
        "publicationTypes",
        "publicationDate",
        "citationCount",
        "referenceCount",
        "abstract",
        "fieldsOfStudy",
    ]
    data = fetch_semantic_scholar_data(arxiv_id, fields)
    paper_info.update(data)
    return paper_info


def parse_arxiv_paper(arxiv_url: str) -> list[Document]:
    """Parse an arXiv paper from a URL."""
    tmp_file_name = f"/tmp/arxiv_{arxiv_url.split('/')[-1].replace('.', '_')}.pdf"
    request.urlretrieve(arxiv_url, tmp_file_name)
    loader = GenericLoader.from_filesystem(
        "/tmp/",
        glob="arxiv_*",
        suffixes=[".pdf"],
        parser=GrobidParser(
            segment_sentences=False,
            grobid_server="http://grobid:8070/api/processFulltextDocument",
        ),
    )
    docs = loader.load()
    os.remove(tmp_file_name)

    # Parse
    sections = OrderedDict()
    for doc in docs:
        section_title = doc.metadata["section_title"]
        if section_title not in sections:
            sections[section_title] = Document(
                page_content="", metadata={"section_title": section_title}
            )
        section = sections[section_title]
        section.page_content = f"{section.page_content} {doc.page_content}"

    return list(sections.values())


def strip_list_string(lst: list) -> str:
    """Remove [, ], ', " from the list."""
    str_list = str(lst)
    stripped_str = re.sub(r"[\[\]\'\"]", "", str_list)
    return stripped_str


def add_newlines(text: str, repl: str):
    # 문장 끝을 찾되, 소수점이 뒤따르지 않는 경우만 매칭
    # (?<!\d\.)\. : 숫자와 점(.)이 앞에 오지 않는 점(.)
    # (?=\s|$) : 점(.) 뒤에 공백이나 문자열의 끝이 오는 경우
    pattern = r"(?<!\d\.)\.\s*(?=\s|$)"

    # 매칭된 부분을 repl(점(.)과 개행문자(\n) 등)로 대체
    return re.sub(pattern, repl, text)


if __name__ == "__main__":
    arxiv_id = "1905.13322"
    data = fetch_paper_info_from_url(arxiv_id=arxiv_id)
    pprint(data)
