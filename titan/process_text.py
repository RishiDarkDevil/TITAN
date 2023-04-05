# Required Libraries

# General
from typing import List, Tuple

# Language Processing
from transformers import CLIPTokenizer
from nltk.corpus import stopwords

import nltk

import stanza

# Model
import torch

__all__ = ['PromptHandler']

# Diffusion Model which we use unless anything else specified
DIFFUSION_MODEL_PATH = 'stabilityai/stable-diffusion-2-base'

# treebank-specific POS (XPOS) tags to keep, other POS tagged tokens will not be retaineds
# The POS tags to retain unless anything else specified
KEEP_POS_TAGS = ['NN', 'NNS', 'NNP', 'NNPS']

# Helper Functions for PromptHandler Class Below
# extract parts of speech
def extract_pos(doc):
  parsed_text = list()
  for sent in doc.sentences:
    parsed_sent = list()
    for wrd in sent.words:
      #extract text and pos
      parsed_sent.append((wrd.text, wrd.xpos))
    parsed_text.append(parsed_sent)
  return parsed_text

# extract lemma
def extract_lemma(doc):
  parsed_text = list()
  for sent in doc.sentences:
    parsed_sent = list()
    for wrd in sent.words:
      # extract text and lemma
      parsed_sent.append((wrd.text, wrd.lemma))
    parsed_text.append(parsed_sent)
  return parsed_text

class PromptHandler:
  """
  This class deals with all the Prompt Processing required for extracting objects for further use in the TITAN pipeline
  """
  
  def __init__(
    self, 
    hf_diffusion_model_path: str = DIFFUSION_MODEL_PATH, 
    hf_diffusion_model_subfolder:str = 'tokenizer',
    tokenize_no_ssplit: bool = True, 
    pos_batch_size: int = 6500, 
    keep_pos_tags: List[str] = KEEP_POS_TAGS
    ):
    
    print('Loading Models...', end='')

    # loads the CLIPTokenizer with the configuration same as that used in the Diffusion Model
    # Using Stanza Tokenizer might generate different tokens compared to the CLIP, leading to misalignment in DAAM - Causing Error
    self.tokenizer = CLIPTokenizer.from_pretrained(hf_diffusion_model_path, subfolder=hf_diffusion_model_subfolder)

    nltk.download('stopwords')

    # Stopwords
    STOPWORDS = set(stopwords.words('english'))
    self.stpwords = STOPWORDS

    # downloads the stanza model
    stanza.download('en')
    # loads the text processing pipeline, pretokenized as CLIPTokenizer will do the tokenizer and need not be handled by stanza
    self.nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', tokenize_no_ssplit=tokenize_no_ssplit, tokenize_pretokenized=True, verbose=True, pos_batch_size=pos_batch_size)

    print('Done')

    self.keep_pos_tags = keep_pos_tags
    
  def clean_prompt(self, sentences: List[str]) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """
    Takes in a list of sentences i.e. prompts and returns a tuple of 3 lists which contains tokenized sentences, cleaned sentences, objects in sentences
    """
    print('Tokenizing...', end='')
    # convert the sentences to lower case and tokenizes the sentences to be passed onto Stanza for POS Tagging
    sentences_lc_tokenized = self.tokenizer.batch_decode([[word for word in sent[1:-1]] for sent in self.tokenizer(sentences)['input_ids'] if len(sent) <= self.tokenizer.model_max_length])
    print('Done')

    # stanza accepts only a single string instead of list of strings. So, we have set the tokenize_no_ssplit=True and have to join each sentence with double newline
    sentence_string = "\n\n".join(sentences_lc_tokenized)

    print('POS Tagging and Lemmatizing...', end='')
    # tokenizes, lemmatizes and pos tags the prompt
    with torch.no_grad():
      processed_prompt = self.nlp(sentence_string)
    print('Done')

    print('Processing...', end='')
    # extracts pos tags from the processed_prompt
    pos_tagged_prompt = extract_pos(processed_prompt)
    
    # lemmatized text
    lemmatized_prompt = extract_lemma(processed_prompt)
    
    del processed_prompt
    
    # keep only the noun words, removes stopwords
    fin_prompt = [[word for word, pos_tag in sent if word is not None and ((pos_tag in self.keep_pos_tags) and (word not in self.stpwords) and (word.isalpha()))] for sent in pos_tagged_prompt]
    obj_prompt = [[word_lemma[1] for word_pos, word_lemma in zip(sent_pos, sent_lemma) if (word_lemma[0] is not None and word_lemma[1] is not None) and ((word_pos[1] in self.keep_pos_tags) and ((word_lemma[0] not in self.stpwords) or (word_lemma[1] not in self.stpwords)) and word_lemma[0].isalpha() and word_lemma[1].isalpha())] for sent_pos, sent_lemma in zip(pos_tagged_prompt, lemmatized_prompt)]
    
    del pos_tagged_prompt, lemmatized_prompt
    print('Done')

    return sentences_lc_tokenized, fin_prompt, obj_prompt