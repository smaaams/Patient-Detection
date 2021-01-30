import re
import unicodedata

import contractions

from consts import punctuations


def normalizer(string):
    string = string.replace('&#039;', "'")
    string = string.replace('&quot;', '"')
    # replace contractions with the complete form of them
    string = contractions.fix(string)
    # remove accented characters (replace them with english chars)
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # remove punctuations
    for punctuation in punctuations:
        string = string.replace(punctuation, ' ')
    # remove all numbers
    string = re.sub('[0-9]', ' ', string)
    # replace all white spaces with single space
    string = ' '.join(string.split())
    return string.lower()  # return final string in lower case
