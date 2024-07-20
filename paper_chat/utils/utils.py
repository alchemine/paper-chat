"""Utility"""

import re
import xml.etree.ElementTree as ET

import requests


def fetch_paper_info_from_url(arxiv_url: str) -> dict:
    # Extract arXiv ID from URL
    arxiv_id_match = re.search(r"arxiv\.org/pdf/(\d+\.\d+)", arxiv_url)
    if not arxiv_id_match:
        print("Invalid arXiv URL")
        return None

    arxiv_id = arxiv_id_match.group(1)

    # arXiv API URL
    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

    # Semantic Scholar API URL
    semantic_scholar_url = f"https://api.semanticscholar.org/v1/paper/arXiv:{arxiv_id}"

    paper_info = {"arxiv_url": arxiv_url, "summary": ""}

    try:
        # Get data from arXiv API
        response = requests.get(api_url)
        response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.content)
        ns = {"arxiv": "http://www.w3.org/2005/Atom"}

        # Extract information
        paper_info["title"] = root.find(".//arxiv:entry/arxiv:title", ns).text.strip()
        paper_info["authors"] = [
            author.find("arxiv:name", ns).text
            for author in root.findall(".//arxiv:entry/arxiv:author", ns)
        ]
        paper_info["abstract"] = root.find(
            ".//arxiv:entry/arxiv:summary", ns
        ).text.strip()
        paper_info["published"] = root.find(".//arxiv:entry/arxiv:published", ns).text
        paper_info["arxiv_id"] = arxiv_id

        # Get data from Semantic Scholar API
        response = requests.get(semantic_scholar_url)
        response.raise_for_status()
        semantic_data = response.json()

        # Extract citation count
        paper_info["citation_count"] = semantic_data.get("citationCount")

    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None
    except AttributeError as e:
        print(f"Error extracting data: {e}")
        return None

    return paper_info
