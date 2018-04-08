import spacy
import ujson

from tqdm import tqdm

from machines.data_generator import path_news_train, path_news_val


def news_generator(path, batch_size=64):
    with open(path, 'r') as _in:
        batch_contents = []
        batch_labels = []
        for line in _in:
            article = ujson.loads(line)
            batch_contents.append(' '.join(article['content']))
            batch_labels.append({'cats': {'FAKE': bool(article['label'])}})

            if len(batch_contents) >= batch_size:
                yield batch_contents, batch_labels


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}


def main():
    nlp = spacy.blank('en')

    textcat = nlp.create_pipe('textcat')
    nlp.add_pipe(textcat, last=True)

    textcat.add_label('FAKE')

    print('Training the model...')
    print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))

    # not using GPU :(
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        epochs = 5
        for i in range(epochs):
            losses = {}
            with tqdm() as progress:
                for batch_contents, batch_labels in news_generator(path_news_train):
                    nlp.update(batch_contents, batch_labels, sgd=optimizer, drop=0.2, losses=losses)
                    progress.update()

            with textcat.model.use_params(optimizer.averages):
                val_contents = []
                val_labels = []
                for batch_contents, batch_labels in news_generator(path_news_val):
                    val_contents += batch_contents
                    val_labels += batch_labels

                scores = evaluate(nlp.tokenizer, textcat, val_contents, val_labels)
                print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                      .format(losses['textcat'], scores['textcat_p'], scores['textcat_r'], scores['textcat_f']))

    nlp.to_disk('data/ner_spacy_model')


if __name__ == '__main__':
    main()
