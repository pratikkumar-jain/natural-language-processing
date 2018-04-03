import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

para = "The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written in the Python programming language. It was developed by Steven Bird and Edward Loper in the Department of Computer and Information Science at the University of Pennsylvania.[5] NLTK includes graphical demonstrations and sample data. It is accompanied by a book that explains the underlying concepts behind the language processing tasks supported by the toolkit,[6] plus a cookbook.[7]"
para1 = "As reviewed by self-proclaimed Tranny pornstar and apparently food critic Lucia M. in August of 2010, illusionist Criss Angel's show Believe at the Cirque du Soleil in Las Vegas, Nevada, is so bad that she goes so far as to write off anyone who would ever go to see the show. Perhaps next time her trick can choose a better show that is less self indulgent cat s**t."

stop_words = set(stopwords.words('english'))

for w in sent_tokenize(para1):
    word = word_tokenize(w)
    tag = pos_tag(word)

    name_entity = nltk.ne_chunk(tag, binary=False)

    name_entity.draw()