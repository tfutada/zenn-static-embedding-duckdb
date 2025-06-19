"""
Do the same as livedoor.py, but using generator functions to yield documents one by one.
"""
import json
import datetime
from typing import Generator, Dict, Any
from pathlib import Path
import random
import consts


def read_document(path: Path) -> Dict[str, Any]:
    """Process a single document and return a dictionary."""
    with path.open() as f:
        lines = [line.rstrip() for line in f]

    if len(lines) < 3:
        return {}

    try:
        dt = datetime.datetime.strptime(lines[1], "%Y-%m-%dT%H:%M:%S%z")
        created_at = int(round(dt.timestamp()))
    except ValueError:
        return {}

    body = ' '.join(lines[2:])

    return {
        "url": lines[0],
        "publisher": path.parent.name,
        "created_at": created_at,
        "body": body
    }


def iter_documents_from_livedoor() -> Generator[Dict[str, Any], None, None]:
    """Yield each document from the corpus as a dictionary."""
    corpus = list(Path(consts.CORPUS_DIR).rglob('*-*.txt'))
    random.shuffle(corpus)

    for path in corpus:
        doc = read_document(path)
        if doc:  # skip empty or failed parses
            yield doc


def write_documents_to_jsonl(output_path: str) -> None:
    """Write documents one by one in JSONL format."""
    with open(output_path, 'w', encoding="utf-8") as fp:
        for doc in iter_documents_from_livedoor():
            json.dump(doc, fp, ensure_ascii=False)
            fp.write('\n')


if __name__ == '__main__':
    write_documents_to_jsonl(consts.LIVEDOOR_JSON)
