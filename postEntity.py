# Brandon V. Syiem
# 6/10/2021

# class to represent a post from a social media site
# params
# id
# title
# content
# title_embedding
# content_embedding

class Post(object):

    def __init__(self, id = -1, title = None, content = None, title_embedding = None, content_embedding = None):
        self.id = id
        self.title = title
        self.content = content
        self.title_embedding = title_embedding
        self.content_embedding = content_embedding
