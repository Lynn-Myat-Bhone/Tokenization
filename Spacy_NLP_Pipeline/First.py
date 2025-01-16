import spacy

with open('./holy_grail.txt', 'r') as file:
    holy_grail = file.read()
    
nlp = spacy.load('en_core_web_sm') # load the pretrain model
preprocess = nlp(holy_grail)

# print([(token.text,token.lemma_) for token in preprocess])

#POS tagging
for token in preprocess:
    print(f'Text:{token.text}, POS Tagging:{token.pos_}') #for entity_type use token.ent_type_
    
"""
Example from DataCamp
 
 # Create a list to store sentences of each Doc container in documents
sentences = [[sent for sent in doc.sents] for doc in documents]

# Create a list to track number of sentences per Doc container in documents
num_sentences = [len([sent for sent in doc.sents]) for doc in documents]
print("Number of sentences in documents:\n", num_sentences, "\n")

# Record entities text and corresponding label of the third Doc container
third_text_entities = [(ent.text, ent.label_) for ent in documents[2].ents]
print("Third text entities:\n", third_text_entities, "\n")

# Record first ten tokens and corresponding POS tag for the third Doc container
third_text_10_pos = [(token.text, token.pos_) for token in documents[2]][:10]
print("First ten tokens of third text:\n", third_text_10_pos)

 
 """   
