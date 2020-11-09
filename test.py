#%%

from functions import *

#%%
sample1 = Sample('Text_Summarization')
result1 = text_summarization(text=sample1)
print(result1)

#%%
sample = Sample('Text_Classification')
result = text_classification(sample)
print(result)

#%%
sample = Sample('Text_Translation')
result = text_translate(text=sample, language='fr')
print(result)

#%%
# TODO change
sample = Sample('Text_Spelling_Correction')
result = text_spelling_correction(text=sample)
print(result)

#%%
sample = Sample('Sentiment_Analysis')
result = text_sentiment_analysis(text=sample)
print(result)

#%%
sample = Sample('Synonymous_Substitution')
result = text_synonymous_substitution(text=sample)
print(result)

#%%
sample = Sample('Grammar_Check')
result = text_grammar_check(text=sample)
print(result)

#%%
sentence1, sentence2 = Sample('Sentence_Similarity')
result = text_sentence_similarity(sentence1, sentence2)
print(result)

#%%
sample = Sample('Text_Score')
result = text_score(text=sample)
print(result)

#%%
sample = Sample('Text_Statistics')
result1 = text_statistics_wordTag(sample)
print(result1)
