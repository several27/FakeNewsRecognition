import numpy as np
import ujson
from tqdm import tqdm

from machines.data_generator import load_fasttext, path_fasttext_jsonl


def main():
    fasttext_dict = load_fasttext()
    with tqdm() as progress:
        with open(path_fasttext_jsonl, 'w') as out_fasttext:
            for word, embedding in fasttext_dict.items():
                embedding = embedding  # type: np.ndarray
                out_fasttext.write(ujson.dumps({'word': word, 'embedding': embedding.tolist()}) + '\n')
                progress.update()


if __name__ == '__main__':
    main()
