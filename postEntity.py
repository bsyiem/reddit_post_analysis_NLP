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

    def __init__(self, id = -1, title = None, content = None, title_embedding = None, content_embedding = None, title_number_chars = 0, content_number_chars = 0):
        self.id = id
        self.title = title
        self.content = content
        self.title_embedding = title_embedding
        self.content_embedding = content_embedding
        self.title_number_chars = title_number_chars
        self.content_number_chars = content_number_chars

    def to_dict(self):
        return {'id': self.id, 'title': self.title, 'content': self.content, 'title_embedding': self.title_embedding, 'content_embedding': self.content_embedding, 'title_number_chars': self.title_number_chars, 'content_number_chars': self.content_number_chars}
