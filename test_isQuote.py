#/usr/bin/python

from __future__ import print_function

# general imports
import sys
import numpy as np
import json
import os
import re
import fastText
# specific
from os.path import exists


def make_gibberish_numbers(n_items, m_numbers_per_item=1):
  digits = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
  teens = ["ten", "eleven", "twelve"] + [i + "teen" for i in ["thir", "four", "fif", "six", "seven", "eigh", "nine"]]
  tens = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
  filler_words = ["point", "dot"]

  months = ["jan", "feb", "march", "april", "may", "june", "july", "aug", "augie", "sept", "oct", "nov", "novie", "dec"]
  date = [
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "nineth", "tenth", "eleventh",
    "twelfth"
  ] + [i + "teenth" for i in ["thir", "four", "fif", "six", "seven", "eigh", "nine"]]
  date += [
    "twenty" + i
    for i in ["", " first", " second", " third", " fourth", " fifth", " sixth", " seventh", " eighth", " nineth"]
  ]
  date += ["thirty" + i for i in ["", " first"]]

  def random_number():
    s = ""
    if np.random.random() < 0.1:
      return np.random.choice(months) + " " + np.random.choice(date)
    r = np.random.random()
    if r < 0.4:
      return np.random.choice(digits)
    elif r < 0.8:
      return np.random.choice(teens)
    else:
      return np.random.choice(tens) + " " + np.random.choice(digits + [""])

  data = []
  for i in range(n_items):
    s = ""
    for j in range(m_numbers_per_item):
      if np.random.random() < 0.5:
        s += random_number() + " " + np.random.choice(filler_words) + " " + random_number()
      else:
        s += random_number()
      if j < (m_numbers_per_item - 1):
        s += " "
    data.append(s)

  return data


class clf():
  model = None
  predict = lambda x: 0

  def __init__(self, model):
    self.model = model
    self.predict = lambda x: [float(_ == "__label__quote") for _ in np.array(self.model.predict(x)[0]).flatten()]


model = clf(fastText.load_model("full_model_quote_or_not.ftz"))

with open("../data/quote_or_not/quotes.txt") as f:
  quotes = [" ".join(_.split()[1:]) for _ in f.read().splitlines()]
  quotes = list(map(lambda i: re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', i), quotes))

not_quotes = make_gibberish_numbers(len(quotes), 14)

# Test quotes and nonquotes without padding
print("{:.2f}% false negative rate on actual quotes".format(100.0 * (1.0 - np.mean(model.predict(quotes)))))
print("{:.2f}% false positive rate on gibberish with numbers".format(100.0 * (np.mean(model.predict(not_quotes)))))

# get a lot of english words
# from http://www.ef.edu/english-resources/english-vocabulary/top-1000-words/
general_english_words = os.popen("cat ../data/quote_or_not/words.txt").read().split()

# pad by up to 10 extra words on each side
padding = 41
pre = [" ".join(np.random.choice(general_english_words, np.random.choice(range(padding)))) for i in range(len(quotes))]
post = [" ".join(np.random.choice(general_english_words, np.random.choice(range(padding)))) for i in range(len(quotes))]
positive_test_cases = [" ".join(i) for i in zip(pre, quotes, post)]

res = model.predict(positive_test_cases)
print("{:.2f}% false negative rate on quote containing phrases".format(100.0 * (1.0 - np.mean(res))))

pre = [
  " ".join(np.random.choice(general_english_words, np.random.choice(range(padding)))) for i in range(len(not_quotes))
]
post = [
  " ".join(np.random.choice(general_english_words, np.random.choice(range(padding)))) for i in range(len(not_quotes))
]
negative_test_cases = [" ".join(i) for i in zip(pre, not_quotes, post)]

res2 = model.predict(negative_test_cases)

print("{:.2f}% false positive rate on gibberish with numbers in longer phrases".format(100.0 * (np.mean(res2))))
