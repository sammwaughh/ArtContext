#!/usr/bin/env python3
"""
Module for ranking candidate sentences and extracting a ~100-word visual description.
It uses Sentence-BERT (SBERT) for unsupervised semantic similarity ranking,
with a bonus for sentences containing visual depiction verbs. The module then
concatenates the top-ranked sentences until a target word count is reached.
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load Sentence-BERT model (using a lightweight model for efficiency).
model = SentenceTransformer("all-MiniLM-L6-v2")

# List of visual depiction verbs to boost relevance.
VISUAL_CONTENT_VERBS = [
    "depict",
    "depicts",
    "portray",
    "portrays",
    "show",
    "shows",
    "illustrate",
    "illustrates",
    "represent",
    "represents",
    "exhibit",
    "exhibits",
]


def rank_sentences(query, candidate_sentences, top_n=10):
    """
    Encode the query and candidate sentences, compute cosine similarities,
    and add a bonus for sentences containing visual-content verbs.
    Returns the top_n sentences along with their similarity scores.
    """
    if not candidate_sentences:
        return [], []
    query_embedding = model.encode(query, convert_to_tensor=True)
    sentence_embeddings = model.encode(candidate_sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
    cosine_scores = cosine_scores.cpu().numpy()
    # Bonus for sentences with visual verbs.
    for i, sent in enumerate(candidate_sentences):
        if any(verb in sent.lower() for verb in VISUAL_CONTENT_VERBS):
            cosine_scores[i] += 0.05
    top_indices = np.argsort(cosine_scores)[::-1][:top_n]
    top_sentences = [candidate_sentences[i] for i in top_indices]
    top_confidences = [round(float(cosine_scores[i]), 2) for i in top_indices]
    return top_sentences, top_confidences


def extract_visual_description(
    query, candidate_sentences, target_word_count=100, top_n=10
):
    """
    Extract a visual description of approximately target_word_count words.
    Ranks candidate sentences using rank_sentences, concatenates the top-ranked sentences,
    and then truncates the result to the target word count.
    """
    ranked_sentences, confidences = rank_sentences(
        query, candidate_sentences, top_n=top_n
    )
    if not ranked_sentences:
        return ""
    description = " ".join(ranked_sentences)
    words = description.split()
    if len(words) > target_word_count:
        description = " ".join(words[:target_word_count])
    return description
