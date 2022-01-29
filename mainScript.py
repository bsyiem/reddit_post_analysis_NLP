# Brandon V. Syiem
# 6/10/2021

# I'll take a very simple approach to this problem by using the post's body as our corpus and the themes as query strings
# may be useful to do the same for title and see if there is a differnce between the mappings

# we could use a semantic search or maybe clustering
# semantic search: https://www.sbert.net/examples/applications/semantic-search/README.html

# random idea:
# clustering first with k = number of themes
# use similarity between theme/query and centroid of clusters to assign cluster to a theme

import sys
import os.path
sys.path.append("..")

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
from io import StringIO

from postEntity import Post
from textProcessor import TextProcessor
from redditAnalysis import RedditAnalysis


pretrained_bert_model_name = 'all-mpnet-base-v2'
textProcessor = TextProcessor(pretrained_bert_model_name)

def getPostEntities(filename):
    p_id = 1

    with open(filename, mode = 'r', encoding = 'utf-8') as file:
        df = pd.read_csv(file, header = 0, keep_default_na = False)

    posts = dict()
    for index, row in df.iterrows():
        post = Post()
        post.id = p_id
        post.title = str(row['title'])
        post.content = str(row['selftext'])
        print(post.title)
        # post.title_embedding = textProcessor.getSentenceEmbedding("") if post.title == None else textProcessor.getSentenceEmbedding(post.title)
        # post.content_embedding = textProcessor.getSentenceEmbedding("") if post.content == None else textProcessor.getSentenceEmbedding(post.content)
        post.title_embedding = textProcessor.getSentenceEmbedding(post.title)
        post.content_embedding = textProcessor.getSentenceEmbedding(post.content)
        post.title_number_chars = len(post.title)
        post.content_number_chars = len(post.content)
        posts[post.id] = post
        p_id += 1

    return posts

# turn embeddings that are read as string to np arrays
def toNpArray(emb_str):
    emb_str = emb_str.replace('[', '')
    emb_str = emb_str.replace(']', '')
    emb_str = emb_str.replace('  ',' ')
    emb_str = emb_str.replace('\n','')
    return np.genfromtxt(StringIO(emb_str), delimiter = ' ') #throws error, not sure how to fix, this is because the delimiter is either a single space in case the proceeding number is negative or double spaces, if the proceeding number is postitive



def main(argv):

    ### loading main dataset
    data_filepath = "data/results-2019-titleandbodyonly-utf8-excel.csv"
    bert_embedding_filepath = "generated/bert_output.csv"

    if(not os.path.exists(bert_embedding_filepath)):
        posts = getPostEntities(data_filepath)
        # fields = ['id', 'title', 'title_embedding', 'content_embedding']
        reddit_bert_df = pd.DataFrame([posts[id].to_dict() for id in posts])
        reddit_bert_df.to_csv(bert_embedding_filepath)

    reddit_bert_df = pd.read_csv(bert_embedding_filepath, keep_default_na = False) #read csv reads the ndarray as a string


    # this is fucked - may as well just re-encode everything
    reddit_bert_df['title_embedding'] = list(map(toNpArray,reddit_bert_df['title_embedding']))
    reddit_bert_df['content_embedding'] = list(map(toNpArray,reddit_bert_df['content_embedding']))


    reddit_bert_df['title_embedding'] = np.array(reddit_bert_df['title_embedding'])
    reddit_bert_df['content_embedding'] = np.array(reddit_bert_df['content_embedding'])


    ### loading themes to perform sentiment analysis
    theme_filepath = "data/investigation-themes-with-additional.csv"
    theme_embedding_filepath = "generated/theme_embedding.csv"

    if(not os.path.exists(theme_embedding_filepath)):
        with open(theme_filepath, mode = 'r', encoding = 'utf-8') as file:
            theme_df = pd.read_csv(file, header = 0, keep_default_na = False)
            theme_df['theme_embedding'] = list(map(textProcessor.getSentenceEmbedding, theme_df['theme']))
            theme_df.to_csv(theme_embedding_filepath)

    theme_df = pd.read_csv(theme_embedding_filepath, keep_default_na = False) #read csv reads the ndarray as a string


    # this is fucked - may as well just re-encode everything
    theme_df['theme_embedding'] = list(map(toNpArray,theme_df['theme_embedding']))

    theme_df['theme_embedding'] = np.array(theme_df['theme_embedding'])

    # run some analysis
    redditAnalysis = RedditAnalysis(textProcessor, reddit_bert_df, theme_df)
    redditAnalysis.semanticSearchK(int(argv[0]))


if __name__ == '__main__':
    main(sys.argv[1:])
