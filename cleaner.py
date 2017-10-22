import sys

from multiprocessing import Pool
from time import time
from markdown import markdown
from tqdm import tqdm
import re
from html.parser import HTMLParser


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ' '.join(self.fed)  # merge with space


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def cleaner_batch_perform(text):
    try:
        cleaner = Cleaner(text)
        cleaner.perform()
        return cleaner.cleared_text
    except Exception:
        print('Error occurred with a string %s' % (text,))
        return ''


class Cleaner:
    text = ""
    cleared_text = ""

    clearing_functions = []

    left_space_characters = {'(', '['}
    right_space_characters = {'.', ',', '!', '?', ':', ';', ')', ']'}  # what about ' and "
    space_characters = {'-', '+', '=', '&', '#'}

    times = {}

    def __init__(self, text: str):
        self.text = text

        self.clearing_functions = [
            self.strip_links,
            # self.strip_markdown,
            self.strip_html,
            self.strip_latex,
            self.strip_newlines,
            self.norm_spaces,
            self.norm_quotes,
        ]

    def perform(self):
        self.cleared_text = self.text

        for function in self.clearing_functions:
            # start = time()
            self.cleared_text = function()

            # if function.__name__ not in self.times:
            #     self.times[function.__name__] = 0
            #
            # self.times[function.__name__] += time() - start

    @staticmethod
    def batch_perform(texts):
        bar = tqdm(total=len(texts))
        cleared = []
        pool = Pool(processes=8, maxtasksperchild=1)
        for text in pool.imap(cleaner_batch_perform, texts, chunksize=100 * 1000):
            cleared.append(text)
            bar.update()

        return cleared

    def lower_case(self) -> str:
        return self.cleared_text.lower()

    def strip_html(self) -> str:
        return strip_tags(self.cleared_text)

    def strip_latex(self) -> str:
        stripped = re.sub("\$\$[^$$]+\$\$", ' ', self.cleared_text)  # $$...$$
        stripped = re.sub("\$[^$]+\$", ' ', stripped)  # $...$
        self.cleared_text = re.sub(r'\\begin\{(.*?)\}(.*?)\\end\{\1\}', ' ', stripped)  # \begin...\end
        # self.cleared_text = re.sub('\[([^]]+)\]', '', stripped)  # [...]
        return self.cleared_text

    def strip_links(self) -> str:
        return re.sub(r'^https?:\/\/.*[\r\n]*', '', self.cleared_text, flags=re.MULTILINE)

    def strip_markdown(self) -> str:
        return markdown(self.cleared_text)

    def strip_newlines(self) -> str:
        return self.cleared_text.replace('\\n', ' ') \
            .replace('\\r', ' ') \
            .replace('\n', ' ') \
            .replace('\r', ' ') \
            .replace('\\', ' ')

    def norm_spaces(self) -> str:
        text = self.cleared_text

        for character in self.left_space_characters:
            text = text.replace(character, ' ' + character)

        for character in self.right_space_characters:
            text = text.replace(character, character + ' ')

        for character in self.space_characters:
            text = text.replace(character, ' ')

        return re.sub('\s+', ' ', text).strip()

    def norm_quotes(self) -> str:
        return self.cleared_text.replace('\'\'', '\'') \
            .replace('""', '"')
