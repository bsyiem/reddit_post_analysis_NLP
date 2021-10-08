from sentence_transformers import SentenceTransformer, util
import torch
from textProcessor import TextProcessor
import numpy as np
import pandas as pd

# Has a known issue where empty strings are also considered similar - not sure how to fix.
# best is to ignore a suggestion with an empty string or NaN
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


            temp_dict = {"score": content_top_cos[0].numpy(), "index": content_top_cos[1].numpy()}
            content_top_cos_df = pd.DataFrame(temp_dict)

            temp_dict = {"score": title_top_cos[0].numpy(), "index": title_top_cos[1].numpy()}
            title_top_cos_df = pd.DataFrame(temp_dict)

            output_path_content = "output/content/"+row['theme']+".csv"
            output_path_title = "output/title/"+row['theme']+".csv"

            # The index starts from 0 so the actual row is +1 (as we have to account for header)
            content_top_cos_df.to_csv(output_path_content)
            title_top_cos_df.to_csv(output_path_title)

            # print(content_top_cos)
            # print(content_top_ang)
            #
            # print(title_top_cos)
            # print(title_top_ang)
