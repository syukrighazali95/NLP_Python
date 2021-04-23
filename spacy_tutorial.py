import spacy

text = """
Dave watched as the forest burned up on the hill,
only a few miles from his house. The car had
been hastily packed and Marta was inside trying to round
up the last of the pets. "Where could she be?" he wondered
as he continued to wait for Marta to appear with the pets.
"""

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
print(doc)
token_list = [token for token in doc]
print(token_list)

filtered_token = [token for token in doc if not token.is_stop]
print(filtered_token)