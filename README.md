# CuisineRAG (South Asian Cuisine)

This repository contains the **data scraping** and **chunking** stages for a South Asian cuisine RAG coursework project.

## Scope
- Source: Wikipedia only
- Current implemented stages:
  - `scraping.ipynb`
  - `chunking.ipynb`

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Order
1. Run all cells in `scraping.ipynb`
2. Run all cells in `chunking.ipynb`

## Outputs
After `scraping.ipynb`:
- `data/raw/south_asian_corpus.json`
- `data/raw/crawl_log.json`

`chunking.ipynb` reads from:
- `data/raw/south_asian_corpus.json`

and compares 3 chunking strategies:
- `CharacterTextSplitter`
- `RecursiveCharacterTextSplitter`
- `MarkdownHeaderTextSplitter`

## Notes
- Wikipedia API user-agent is configured as:
  - `CuisineRAG/1.0 (mingyi.jin@student.manchester.ac.uk)`
- If `chunking.ipynb` reports missing corpus file, run `scraping.ipynb` first.
