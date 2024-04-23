from __future__ import print_function
import os.path as op
from random import Random
from math import log10
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Constants for special tokens
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Python 2 / 3 compatibility helpers
unicode_type = type(u"")
bytes_type = type(b"")

english_digit_names = [
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
]

english_ten_names = [
    'ten',
    'twenty',
    'thirty',
    'forty',
    'fifty',
    'sixty',
    'seventy',
    'eighty',
    'ninety',
]

english_exceptions = {
    11: 'eleven',
    12: 'twelve',
    13: 'thirteen',
    14: 'fourteen',
    15: 'fifteen',
    16: 'sixteen',
    17: 'seventeen',
    18: 'eighteen',
    19: 'nineteen',
}

def to_english_phrase(number):
    if number >= int(1e6):
        raise NotImplemented('number should be less than a million, got %s' % number)
    thousands, thousand_remainder = divmod(number, 1000)
    hundreds, hundred_remainder = divmod(thousand_remainder, 100)
    tens, units = divmod(hundred_remainder, 10)

    words = []
    if thousands > 0:
        words += to_english_phrase(thousands).split() + ['thousand']
    if hundreds > 0:
        words += [english_digit_names[hundreds - 1], 'hundred']

    if hundred_remainder in english_exceptions:
        words.append(english_exceptions[hundred_remainder])
    else:
        if tens > 0:
            words.append(english_ten_names[tens - 1])
        if units > 0:
            words.append(english_digit_names[units - 1])

    return " ".join(words)

def generate_translations(low=1, high=int(1e6) - 1, exhaustive=int(1e4),
                          random_seed=0):
    """Generate random numbers and their french translations

    Try to balance large numbers logarithmically so that there are many
    samples between 1e4 an 1e5 than 1 and 'exhaustive'.
    """
    exhaustive = min(exhaustive, high)
    numbers, french_numbers = [], []
    # generate all the small numbers: this can
    # help make learning easier by over-representing the simple
    # cases (curriculum learning)
    for x in range(low, exhaustive + 1):
        numbers.append(str(x))
        french_numbers.append(to_english_phrase(x))

    rng = Random(random_seed)

    # logarithmically balanced sampling beyond exhaustive enumeration
    i = int(log10(exhaustive))
    for i in range(int(log10(exhaustive)), int(log10(high)) + 1):
        local_low = 10 ** i
        local_high = min(10 ** (i + 1), high)
        for _ in range(exhaustive):
            x = rng.randint(local_low, local_high)
            numbers.append(str(x))
            french_numbers.append(to_english_phrase(x))
    numbers, french_numbers = shuffle(numbers, french_numbers,
                                      random_state=random_seed)
    return numbers, french_numbers


def prepare_datasets(data_dir, low=0, high=int(1e6) - 1,
                     exhaustive=int(1e4), random_seed=0):
    """Generate training and development translated sentences.

    Store the resulting sentences and vocabulary in the data_dir folder.
    """
    fr_train_token = op.join(data_dir, "fr_train.token")
    fr_train_text = op.join(data_dir, "fr_train.txt")
    fr_dev_token = op.join(data_dir, "fr_dev.token")
    fr_dev_text = op.join(data_dir, "fr_dev.txt")
    fr_vocab_path = op.join(data_dir, "fr.vocab")

    digits_train_token = op.join(data_dir, "digits_train.token")
    digits_train_text = op.join(data_dir, "digits_train.txt")
    digits_dev_token = op.join(data_dir, "digits_dev.token")
    digits_dev_text = op.join(data_dir, "digits_dev.txt")
    digits_vocab_path = op.join(data_dir, "load_vocabulary")

    filenames = [fr_train_token, digits_train_token,
                 fr_dev_token, digits_dev_token,
                 fr_vocab_path, digits_vocab_path]

    if all(op.exists(fn) for fn in filenames):
        print("Reusing existing number translation dataset from %s."
              % data_dir)
        return filenames

    print("Generating number translation dataset in %s..."
          % data_dir)
    numbers, french_numbers = generate_translations(
        low=low, high=high, exhaustive=exhaustive, random_seed=random_seed)
    num_train, num_dev, fr_train, fr_dev = train_test_split(
        numbers, french_numbers, test_size=0.1, random_state=random_seed)

    # build the vocabularies from the training set only to get a chance
    # to have some out-of-vocabulary words in the dev set.
    fr_vocab, rev_fr_vocab = build_vocabulary(fr_train, word_level=True)
    num_vocab, rev_num_vocab = build_vocabulary(num_train, word_level=False)

    save_vocabulary(rev_fr_vocab, fr_vocab_path)
    save_vocabulary(rev_num_vocab, digits_vocab_path)

    save_sentences(fr_train, fr_vocab, fr_train_token, fr_train_text)
    save_sentences(fr_dev, fr_vocab, fr_dev_token, fr_dev_text)

    save_sentences(num_train, num_vocab, digits_train_token,
                   digits_train_text, word_level=False)
    save_sentences(num_dev, num_vocab, digits_dev_token,
                   digits_dev_text, word_level=False)

    return filenames


def tokenize(sentence, word_level=True):
    if word_level:
        return sentence.split()
    else:
        return [sentence[i:i + 1] for i in range(len(sentence))]


def save_sentences(sentences, vocab, token_filename, text_filename,
                   word_level=True, encoding='utf-8'):
    with open(token_filename, 'wb') as token_f, \
            open(text_filename, 'wb') as text_f:
        for sentence in sentences:
            if isinstance(sentence, unicode_type):
                sentence = sentence.encode(encoding)
            text_f.write(sentence)
            text_f.write(b'\n')
            tokens = tokenize(sentence, word_level=word_level)
            token_ids = [vocab.get(t, UNK_ID) for t in tokens]
            token_f.write(b" ".join(str(t).encode(encoding)
                                    for t in token_ids))
            token_f.write(b"\n")


def save_vocabulary(rev_vocab, filename, encoding='utf-8'):
    with open(filename, 'wb') as f:
        for token in rev_vocab:
            f.write(token.encode(encoding))
            f.write(b'\n')


def load_vocabulary(filename, encoding='utf-8'):
    vocab, rev_vocab = {}, []
    with open(filename, 'rb') as f:
        for line in f:
            rev_vocab.append(line.decode(encoding).strip())
    for i, token in enumerate(rev_vocab):
        vocab[token] = i
    return vocab, rev_vocab


def build_vocabulary(sentences, word_level=True):
    """Extract a sorted vocabulary from a set of sentences

    Word level (split on whitespaces and lexicographic order):

    >>> sentences = ['un deux trois', 'cinq', 'sept trois']
    >>> vocabulary, rev_vocabulary = build_vocabulary(
    ...     sentences, word_level=True)
    >>> len(vocabulary)
    9
    >>> sorted(vocabulary.items(), key=lambda x: x[1])
    ...                            # doctest: +NORMALIZE_WHITESPACE
    [('_PAD', 0), ('_GO', 1), ('_EOS', 2), ('_UNK', 3),
     ('cinq', 4), ('deux', 5), ('sept', 6), ('trois', 7), ('un', 8)]
    >>> rev_vocabulary
    ...                            # doctest: +NORMALIZE_WHITESPACE
    ['_PAD', '_GO', '_EOS', '_UNK', 'cinq', 'deux', 'sept', 'trois', 'un']

    """
    rev_vocabulary = START_VOCAB[:]
    unique_tokens = set()
    for sentence in sentences:
        tokens = tokenize(sentence, word_level=word_level)
        unique_tokens.update(tokens)
    rev_vocabulary += sorted(unique_tokens)
    vocabulary = {}
    for i, token in enumerate(rev_vocabulary):
        vocabulary[token] = i
    return vocabulary, rev_vocabulary


def sentence_to_token_ids(sentence, vocabulary, word_level=True):
    """Convert a string to a sequence of integer token ids

    The meaning of the integer codes depends on the vocabulary.
    Unknown tokens are replaced by the _UNK code.

    >>> UNK_ID
    3
    >>> vocabulary = {'_UNK': 3, 'un': 4, 'deux': 5, 'trois': 6}
    >>> sentence_to_token_ids('un quatre trois', vocabulary)
    [4, 3, 6]

    >>> vocabulary = {'_UNK': 3, '1': 4, '2': 5, '3': 6}
    >>> sentence_to_token_ids('143', vocabulary, word_level=False)
    [4, 3, 6]
    """
    tokens = tokenize(sentence, word_level=word_level)
    return [vocabulary.get(token, UNK_ID) for token in tokens]


def token_ids_to_sentence(token_ids, rev_vocabulary, word_level=True):
    """Decode a sequence back to its string representation

    >>> rev_vocabulary = ['_PAD', '_GO', '_EOS', '_UNK', '1', '2', '3']
    >>> token_ids = [5, 6, 4, 2, 0, 0, 0, 0, 0]
    >>> token_ids_to_sentence(token_ids, rev_vocabulary, word_level=False)
    '231'

    >>> rev_vocabulary = ['_PAD', '_GO', '_EOS', '_UNK', 'un', 'deux']
    >>> token_ids = [5, 3, 5, 2, 0, 0, 0, 0, 0]
    >>> token_ids_to_sentence(token_ids, rev_vocabulary)
    'deux _UNK deux'

    """
    sep = " " if word_level else ""
    return sep.join(rev_vocabulary[idx] for idx in token_ids if idx > 2)