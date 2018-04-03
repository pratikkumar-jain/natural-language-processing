from nltk.corpus import wordnet

synonyms = wordnet.synsets("clear")

print(synonyms)  # to print all the sysets

list_words = []
for i in range(0, len(synonyms)-1):
    for j in range(len(synonyms[i].lemmas())):
        list_words.append(synonyms[i].lemmas()[j].name())

set_words = set(list_words)
print(set_words)

for i in range(len(synonyms)):
    print('WORD: {}'.format(synonyms[i].name()))
    print('DEFN: {}'.format(synonyms[i].definition()))
    print('EG: {}'.format(synonyms[i].examples()))


synonym_word = []
antonym_word = []

for syn in wordnet.synsets("nice"):
    for l in syn.lemmas():
        synonym_word.append(l.name())
        if l.antonyms():
            antonym_word.append(l.antonyms()[0].name())


print(set(synonym_word))
print(set(antonym_word))


w1 = wordnet.synsets('truck')
w2 = wordnet.synsets('car')
w3 = wordnet.synsets('airplane')
w4 = wordnet.synsets('ant')

print(w1[0].wup_similarity(w2[0]))
print(w1[0].wup_similarity(w3[0]))
print(w1[0].wup_similarity(w4[0]))
