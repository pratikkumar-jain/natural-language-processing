from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

para = "The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written in the Python programming language. It was developed by Steven Bird and Edward Loper in the Department of Computer and Information Science at the University of Pennsylvania.[5] NLTK includes graphical demonstrations and sample data. It is accompanied by a book that explains the underlying concepts behind the language processing tasks supported by the toolkit,[6] plus a cookbook.[7]"

stop_words = set(stopwords.words('english'))

words = [w for w in word_tokenize(para) if w not in stop_words]
 
print(words)
