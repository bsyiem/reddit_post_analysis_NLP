# Brandon V. Syiem
# 6/10/2021

# I'll take a very simple approach to this problem by using the post's body as our corpus and the themes as query strings
# may be useful to do the same for title and see if there is a differnce between the mappings

# we could use a semantic search or maybe clustering
# semantic search: https://www.sbert.net/examples/applications/semantic-search/README.html

import sys
import os.path
sys.path.append("..")

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

from postEntity import Post
from textProcessor import TextProcessor


pretrained_bert_model_name = 'all-mpnet-base-v2'
textProcessor = TextProcessor(pretrained_bert_model_name)

def getPostEntities(filename):
    p_id = 1

    # with open(filename, mode = 'r', encoding = 'utf-8') as file:
    #     df = pd.read_excel(file, header = 0) # default sheet index = 0

    with open(filename, mode = 'r', encoding = 'utf-8') as file:
        df = pd.read_csv(file, header = 0)

    posts = dict()
    for index, row in df.iterrows():
        post = Post()
        post.id = p_id
        post.title = str(row['title'])
        post.content = str(row['selftext'])
        print(post.title)
        print(str(len(post.title)) + ":" + str(len(post.content)))
        post.title_embedding = textProcessor.getSentenceEmbedding(" ") if post.title == None else textProcessor.getSentenceEmbedding(post.title)
        post.content_embedding = textProcessor.getSentenceEmbedding(" ") if post.content == None else textProcessor.getSentenceEmbedding(post.content)
        posts[post.id] = post
        p_id += 1

    return posts



def main(argv):

    # filename = "data/results-2019-titleandbodyonly-utf8.csv"
    data_filename = "data/results-2019-titleandbodyonly-utf8-excel.csv"
    bert_embedding_filename = "bert_output.csv"

    if(not os.path.exists(bert_embedding_filename)):
        posts = getPostEntities(data_filename)
        # fields = ['id', 'title', 'title_embedding', 'content_embedding']
        reddit_bert_df = pd.DataFrame([posts[id].to_dict() for id in posts])
        reddit_bert_df.to_csv(bert_embedding_filename)

    reddit_bert_df = pd.read_csv(bert_embedding_filename)
    print(reddit_bert_df)

    # for p_id in posts:
    #     print(posts[p_id].title.encode("utf-8"))



if __name__ == '__main__':
    main(sys.argv[1:])
