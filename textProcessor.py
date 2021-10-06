# Brandon V. Syiem
# 6/10/2021

#adapted from: https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21

import sys
import numpy as np

#spacy dependencies
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en import English
parser = English()

#nltk dependencies
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
english_stopwords = set(nltk.corpus.stopwords.words('english'))

#gensim LDA (Latent Dirichlet Allocation)
import gensim
from gensim import corpora
import pickle

#BERT sentence transformers
from sentence_transformers import SentenceTransformer

#words need to be excludedWords
excludedWords = [""]

class TextProcessor(object):

    def __init__(self, pretrained_bert_model_name):
        self.pretrained_bert_model = SentenceTransformer(pretrained_bert_model_name)

    #get a list of subjects
    def getSubjects(self, sentence):
        doc = nlp(sentence)
        subject_tokens = [token for token in doc if (token.dep_ == "nsubj")]
        return subject_tokens

    #tokenize text and can be used for cleaning
    def cleanAndTokenize(self,text):
        processed_tokens = []
        tokens = parser(text)
        for token in tokens:
            if token.orth_.isspace():
                continue
            elif token.like_url: #primarily for scrapped webpages
                processed_tokens.append('URL')
            elif token.orth_.startswith('@'): #primarily for scrapped webpages
                processed_tokens.append('SCREEN_NAME')
            else:
                processed_tokens.append(token.lower_)
        return processed_tokens

    #returns the root of the word
    def lemmatize(self,word):
        return WordNetLemmatizer().lemmatize(word)
        # # alternative lemmatization
        # lemma = wn.morphy(word)
        # if lemma is None:
        #     return word
        # else:
        #     return lemma

    # prepares the text for LDA by
    # removing words whose length is less than 4
    # removing stopwords
    # lemmatizing the words
    def prepareText(self,text):
        tokens = self.cleanAndTokenize(text)
        #only consider words with length > 4
        tokens = [token for token in tokens if len(token) > 4]
        tokens = [token for token in tokens if token not in english_stopwords]
        tokens = [self.lemmatize(token) for token in tokens]
        tokens = [token for token in tokens if token not in excludedWords]
        return tokens

    def getLDATopics(self,ldaModel, num_print_words):
        topics = ldaModel.print_topics(num_words = num_print_words)
        return topics

    def getLDAModel(self,filenames,number_topics):
        text_data = []
        for filename in filenames:
            file = open(filename,"r")

            for line in file:
                tokens = self.prepareText(line)
                text_data.append(tokens)


        ldaDictionary = corpora.Dictionary(text_data)
        corpus = [ldaDictionary.doc2bow(text) for text in text_data]
        #self.savePreparedDictionary(dictionary,corpus)

        ldaModel = gensim.models.ldamodel.LdaModel(corpus, num_topics = number_topics,id2word = ldaDictionary, passes = 15)
        #ldaModel.save('model5.gensim')

        return ldaModel,ldaDictionary

    def getTopicProbabilities(self, clue, ldaModel, ldaDictionary):
        clue_tokens = self.cleanAndTokenize(clue)
        clue_bow = ldaDictionary.doc2bow(clue_tokens)
        return (ldaModel.get_document_topics(clue_bow))


    #this is probably not needed
    def savePreparedDictionary(self,ldaDictionary,corpus):
        pickle.dump(corpus,open('corpus.pkl','wb'))
        ldaDictionary.save('ldaDictionary.gensim')

    def test(self,filenames, ldaModel, ldaDictionary):
        # prints all recognized topisc
        topics = self.getLDATopics(ldaModel,4)
        for topic in topics:
            print(topic)

        for filename in filenames:
            file =  open(filename,"r")

            #prints probabilities of clue belonging to topic
            for line in file:
                print (line)
                print (self.getTopicProbabilities(line,ldaModel,ldaDictionary))

###############################################
# BERT FUNCTIONS
###############################################
    @classmethod
    def cosine_similarity(self, a, b):
        # Cosine is probably not the best to compare with zero vectors
        # but works in this case as we want no Similarity  (i.e., 0)
        # if there are no selected or non-selected post its (because their embeddings would be 0 vectors)
        if(np.all(np.array(a)) == 0 or np.all(np.array(b)) == 0):
            return 0

        ## TODO: May be return a value outside of [-1,1]
        # np.clip may help
        # return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))
        return np.clip((np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))), -1, 1)

    @classmethod
    def angular_distance(self, a, b):
        return np.arccos(TextProcessor.cosine_similarity(a,b))/np.pi

    @classmethod
    def angular_similarity(self, a, b):
        return 1 - TextProcessor.angular_distance(a, b)

    @classmethod
    def euclideanDistance(self, a, b):
        return np.linalg.norm(a-b)

    def getSentenceEmbedding(self,sentence):
        return self.pretrained_bert_model.encode(sentence)
