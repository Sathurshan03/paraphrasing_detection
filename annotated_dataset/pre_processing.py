"""
This script is used to preprocess the annotation.

It will
    - Score word similarity.
"""

import argparse
import re
from typing import List

import pandas as pd
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def preprocess(text: str) -> List[str]:
    """Tokenizes and normalizes tokens."""
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    stemmed = [stemmer.stem(word) for word in words]
    return stemmed

def similarity(t1: str, t2: str) -> float:
    """Computes the similarity score between two text."""
    words1 = preprocess(t1)
    words2 = preprocess(t2)

    if len(words1) == 0 or len(words2) == 0:
        return 0.0

    set1 = set(words1)
    set2 = set(words2)

    common1 = sum(1 for w in words1 if w in set2)
    common2 = sum(1 for w in words2 if w in set1)

    return 0.5 * ((common1 / len(words1)) + (common2 / len(words2)))


def main():
    """Main entry code."""

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()


    df = pd.read_excel(args.filename)

    # Compute similarity score.
    df.iloc[:, 2] = df.apply(lambda row: similarity(str(row.iloc[0]), str(row.iloc[1])),
                            axis=1)

    # Save score to same file.
    df.to_excel(args.filename, index=False)

if __name__ == "__main__":
    main()
