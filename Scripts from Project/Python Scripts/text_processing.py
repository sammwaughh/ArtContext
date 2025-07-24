#!/usr/bin/env python3
"""
Module for processing markdown text with a robust NLP pipeline.
Includes:
  - Preprocessing markdown (removing images, links, headers)
  - Resolving coreferences using FastCoref (FCoref) to standardize painting references
  - Filtering out non-content lines (e.g., headings, citations)
  - Splitting text into sentences (using spaCy if available, otherwise nltk)
  - Building an expanded query sentence using dynamic synonyms for depicted elements
  - Gathering sentences with a sliding window around painting name occurrences
"""

import re

import nltk
import spacy
from fastcoref import FCoref
from nltk.corpus import wordnet as wn

# Ensure required nltk data is downloaded.
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Optionally load a spaCy model for robust sentence segmentation.
try:
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None

# Initialize FastCoref for coreference resolution.
f_coref_model = FCoref()


def preprocess_markdown(md_content):
    """
    Remove common markdown syntax (images, links, headers) to produce plain text.
    """
    text = re.sub(r"!\[.*?\]\(.*?\)", "", md_content)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n+", "\n", text)
    return text


def resolve_coreferences(text, painting_name):
    """
    Use FastCoref (FCoref) to perform coreference resolution on the text.
    For any coreference cluster that includes a mention of painting_name,
    replace all other mentions in that cluster with painting_name.
    """
    result = f_coref_model.predict(text)
    clusters = result.clusters if hasattr(result, "clusters") else []
    tokens = text.split()
    for cluster in clusters:
        cluster_texts = [" ".join(tokens[start : end + 1]) for start, end in cluster]
        if any(painting_name.lower() in mention.lower() for mention in cluster_texts):
            for start, end in cluster:
                mention_text = " ".join(tokens[start : end + 1])
                if mention_text.lower() != painting_name.lower():
                    pattern = r"\b" + re.escape(mention_text) + r"\b"
                    text = re.sub(pattern, painting_name, text, flags=re.IGNORECASE)
    return text


def filter_non_heading_lines(text):
    """
    Remove lines that are likely non-content (e.g., headings, citations).
    Skips lines with fewer than 5 words or containing tokens like 'fig.', 'table', 'doi', 'http', etc.
    Returns the remaining text as a single string.
    """
    lines = text.split("\n")
    skip_tokens = ["fig.", "figure", "table", "doi", "http", "www", "ISBN"]
    filtered = []
    for line in lines:
        words = line.strip().split()
        if len(words) < 5:
            continue
        if any(tok in line.lower() for tok in skip_tokens):
            continue
        filtered.append(line.strip())
    return " ".join(filtered)


def split_into_sentences(text):
    """
    Split text into sentences.
    Uses spaCy for robust segmentation if available; otherwise, falls back to nltk.
    """
    if _nlp:
        doc = _nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    else:
        return nltk.sent_tokenize(text)


def expand_depicts_item(item):
    """
    Dynamically generate synonyms for a depicted item using WordNet.
    Returns a list of synonyms including the original term.
    """
    synonyms = set()
    for syn in wn.synsets(item):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    synonyms = list(synonyms)
    # Ensure the original term is included.
    if item.lower() not in [s.lower() for s in synonyms]:
        synonyms.append(item)
    return synonyms


def build_expanded_query_sentence(title, depicts):
    """
    Build an expanded query sentence using the painting title and dynamic synonyms for each depicted element.
    Example:
      "Painting 'The Starry Night' shows (sky or blue sky), (raven or black bird), (grass or green field)."
    """
    depicted_elements = [elem.strip() for elem in depicts.split(",") if elem.strip()]
    if not depicted_elements:
        return f"Painting '{title}' shows unspecified details."
    expanded_parts = []
    for elem in depicted_elements:
        synonyms = expand_depicts_item(elem)
        part = " or ".join(synonyms)
        expanded_parts.append(f"({part})")
    expanded_str = ", ".join(expanded_parts)
    return f"Painting '{title}' shows {expanded_str}."


def gather_sentences_with_sliding_window(sentences, painting_name, window=10):
    """
    Return all sentences within a sliding window (±window) around any sentence that contains painting_name.
    """
    indices = [
        i for i, sent in enumerate(sentences) if painting_name.lower() in sent.lower()
    ]
    selected = set()
    for idx in indices:
        start = max(0, idx - window)
        end = min(len(sentences), idx + window + 1)
        for i in range(start, end):
            selected.add(sentences[i].strip())
    return list(selected)
