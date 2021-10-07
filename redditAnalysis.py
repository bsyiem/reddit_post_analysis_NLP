from sentence_transformers import SentenceTransformer, util
import torch
from textProcessor import TextProcessor
import numpy as np
import pandas as pd

class RedditAnalysis(object):

    def __init__(self, textProcessor, data_embedding_df, theme_embedding_df):
        self.textProcessor = textProcessor
        self.data_embedding_df = data_embedding_df
        self.theme_embedding_df = theme_embedding_df

    def semanticSearchK(self, top_k = 10):
        # print(self.theme_embedding_df['theme_embedding'].values)
        # print(self.data_embedding_df['content_embedding'])
        top_k = min(top_k, self.data_embedding_df.shape[0])

        content_emb_tensor = torch.tensor(self.data_embedding_df['content_embedding'])
        title_emb_tensor = torch.tensor(self.data_embedding_df['title_embedding'])

        for index, row in self.theme_embedding_df.iterrows():
            theme_emb_tensor = torch.tensor(row['theme_embedding'])

            content_cos_scores = util.pytorch_cos_sim(theme_emb_tensor, content_emb_tensor)[0]
            content_ang_scores = 1 - (np.arccos(content_cos_scores)/np.pi)

            title_cos_scores = util.pytorch_cos_sim(theme_emb_tensor, title_emb_tensor)[0]
            title_ang_scores = 1 - (np.arccos(title_cos_scores)/np.pi)

            content_top_cos =  torch.topk(content_cos_scores, k = top_k)
            content_top_ang =  torch.topk(content_ang_scores, k = top_k)

            title_top_cos =  torch.topk(title_cos_scores, k = top_k)
            title_top_ang =  torch.topk(title_ang_scores, k = top_k)

            # content_top_cos_df = pd.DataFrame(content_top_cos)
            # title_top_cos_df = pd.DataFrame(title_top_cos)
            #
            # output_path_content = "output/content/"+self.theme_embedding_df['theme']+".csv"
            # output_path_title = "output/title/"+self.theme_embedding_df['theme']+".csv"
            #
            # content_top_cos_df.to_csv(output_path_content)
            # title_top_cos_df.to_csv(output_path_title)


            # The index starts from 0 so the actual row is +1
            print(content_top_cos)
            print(content_top_ang)

            print(title_top_cos)
            print(title_top_ang)
