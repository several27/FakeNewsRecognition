from embeddings.embedding import Embedding
from gensim.models import FastText
from tqdm import tqdm

from machines.data_generator import path_fasttext, path_data


def main():
    print('Loading fasttext...')
    fasttext = FastText.load_fasttext_format(path_fasttext)

    print('Writing')
    batch = []
    e = Embedding()
    e.db = e.initialize_db(path_data + 'fasttext.db')
    for word in tqdm(fasttext.wv.vocab):
        embedding = fasttext[word]
        batch.append((word, embedding))

        if len(batch) > 100:
            e.insert_batch(batch)
            batch = []

    e.insert_batch(batch)


if __name__ == '__main__':
    main()
