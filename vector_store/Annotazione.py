
class Annotazione:
    def __init__(self,riga):
        self.post_id = riga['post_id']
        self.post_date = riga['post_date']
        self.post_topic_keywords = riga['post_topic_keywords']
        self.post_text = riga['post_text']
        self.list_annotatore_A = riga['list_annotatore_A']
        self.list_annotatore_B = riga['list_annotatore_B']

    def __str__(self):
        return f"Post ID: {self.post_id}\n, Date: {self.post_date}\n, Keywords: {self.post_topic_keywords}\n, Text: {self.post_text}\n, Annotations A: {self.list_annotatore_A}\n, Annotations B: {self.list_annotatore_B}\n"