# The following code is an example to compare the
# working of stemmer and Leammatizer in NLTK
# for stemming, PorterStemmer is used
# and for lemmatizing, WordNetLemmatizer is used
# to remove the punctuation, RegexpTokenizer is used
# and to remove the stop words, stopwords is used

from beautifultable import BeautifulTable
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

para = "Even though it's a tiny bit hard to figure out how to order your first time there, the food is super fresh and all made to order. Once you enter the bigger grocery store (housed in the home of an old car dealership), keep walking straight until you get into the bread area and then turn right. Once you're in the taqueria area, scan the menu boards and the picture items on their large electronic boards, make a selection and then get your number. If you want an aquafresca or fruit or tamales, order those as well and then walk to your immediate left to give that station your ticket. The food comes out pretty quickly and is always delicious. It's sort of strange that you can't get any chips with your meal, but they are homemade and come with a side order of guacamole or beans or rice. I've had almost all of the meats in both taco, sope and gordita form and most are really good (especially love the lengua and the pastor). The tacos dorados are especially excellent too and the meat platters are usually solid. Other than a really greasy carne asada and an overly chewy pork chop, everything I've had there has been flawless. And finally don't forget the salsa bar. Escabeche, pico, 3-4 types or salsa, cilantro and onion -- all fresh and all great. The only pain in the butt part is that there are no places to put the salsa bar items unless you either beg the folks behind the counter for little containers or you get your food to go and load up on the less saucy items in that container."

lemmatizer = WordNetLemmatizer()

stemmer = PorterStemmer()

tokenizer = RegexpTokenizer(r'\w+')

table = BeautifulTable()
table.column_headers = ["Actual", "Lemmatize", "Stem"]

words = [w for w in tokenizer.tokenize(
    para) if w not in set(stopwords.words('english'))]

for word in words:
    table.append_row([word, lemmatizer.lemmatize(word), stemmer.stem(word)])

print(table)
