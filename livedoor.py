"""
This script processes text documents from the livedoor news corpus and converts them
into JSON format suitable for import into vector databases such as Qdrant.

- It recursively reads article files (excluding metadata files) from CORPUS_DIR.
- Each article file is parsed to extract the URL, publication timestamp, and body text.
- The parsed data is saved as newline-delimited JSON objects into `livedoor.json`.
- Output JSON includes fields: url, publisher (derived from directory name), created_at (as UNIX timestamp), and body.

Intended for use in downstream tasks such as embedding generation or retrieval-augmented generation (RAG).
"""

import json
import datetime
from typing import List, Dict, Any
from pathlib import Path
import random
import consts


def read_document(path: Path) -> Dict[str, Any]:
    """1ドキュメントの処理"""
    with path.open() as f:
        lines: List[Any] = f.readlines()
        lines = list(map(lambda x: x.rstrip(), lines))

        d = datetime.datetime.strptime(lines[1], "%Y-%m-%dT%H:%M:%S%z")
        created_at = int(round(d.timestamp()))  # 数値(UNIXエポックタイプ)に変換

        # 1行目はURL、2行目は作成日時、3行目以降が本文
        body = ' '.join(lines[2:])  # 本文を結合

        return {
            "url": lines[0],
            "publisher": path.parent.name,
            "created_at": created_at,
            "body": body
        }


def load_dataset_from_livedoor_files() -> None:
    # NB. exclude LICENSE.txt, README.txt, CHANGES.txt
    corpus: List[Path] = list(Path(consts.CORPUS_DIR).rglob('*-*.txt'))
    random.shuffle(corpus)  # 記事をシャッフルします

    with open(consts.LIVEDOOR_JSON, 'w') as fp:
        for x in corpus:
            doc: Dict[str, str] = read_document(x)
            json.dump(doc, fp)  # 1行分
            fp.write('\n')


if __name__ == '__main__':
    load_dataset_from_livedoor_files()
