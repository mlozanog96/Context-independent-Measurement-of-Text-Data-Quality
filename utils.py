#### Context-Independent Mesurement of Text Data Quality

### Libraries and Settings
import pandas as pd
# For Natural Language Processing functions
import spacy
# To find patterns in text
import re
# Removing stop words, words dictionary
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('words')
from nltk.corpus import words
# For preprocessing
from gensim.parsing.preprocessing import remove_stopwords,strip_numeric, strip_punctuation, strip_short
# Identify spelling mistakes and unknown words
from spellchecker import SpellChecker

### Preprocessing function
# Create a list of the stopwords in english
stopwords_en_processing = set(stopwords.words('english'))

# Remove words from stopword list (must be included in text)
words_to_remove_en= ["aren",'again','but','can','couldn', "couldn't",'didn', "don't",'doesn', "doesn't", 'down', "didn't", 'for','hadn',"hadn't",'hasn', "hasn't",'have','haven', "haven't","aren't",'isn', "isn't",'mightn', "mightn't",'mustn', "mustn't",'needn',"needn't",'no', 'nor', 'not','out', 'over','the',"shan't",'shouldn', "shouldn't",'very','wasn',"wasn't",'weren',"weren't", "won't",'wouldn', "wouldn't"]
for word in words_to_remove_en:
    stopwords_en_processing.remove(word)


def processed_text(df,column):
    # Call SpellChecker function
    spell = SpellChecker()
    processed_text=[]
    for value in df[column]:
        # Remove users and links
        preprocessed_text = re.sub(r'@[A-Za-z0-9_]+|http\S+|www.\S+|&+|ð|Ÿ|’|•','',str(value), flags=re.I)
        # Convert contractions
        preprocessed_text = re.sub(r'â€™',"'", preprocessed_text, flags=re.I)
        preprocessed_text = re.sub(r"'m",' am',preprocessed_text, flags=re.I)
        preprocessed_text = re.sub(r"'d",' would',preprocessed_text, flags=re.I)
        preprocessed_text = re.sub(r"can't",' cannot',preprocessed_text, flags=re.I)
        preprocessed_text = re.sub(r"n't",' not',preprocessed_text, flags=re.I)
        preprocessed_text = re.sub(r"'s",' is',preprocessed_text, flags=re.I)
        # Remove punctuation
        preprocessed_text = strip_punctuation(preprocessed_text)
        # Remove numbers
        preprocessed_text = strip_numeric(preprocessed_text)
        # Convert to lower case
        preprocessed_text= preprocessed_text.lower()
        # Correct spelling mistakes        
        ## Tokenize the text into words
        words = preprocessed_text.split()
        # Correct words
        word_correction = [spell.correction(word) for word in words]
        # Concatenated words
        preprocessed_text = [word for word in word_correction if word is not None]
        preprocessed_text =' '.join(preprocessed_text)
        # Remove stopwords 
        preprocessed_text = remove_stopwords(preprocessed_text, stopwords=stopwords_en_processing)
        processed_text.append(preprocessed_text) 
    # Save new information in a new column
    df['processed_text']= processed_text
    return df

### Context-Independent Metrics of Text Data Quality
## Metric 1: Abbreviation metric
# Create a list of the stopwords in english
stopwords_en = set(stopwords.words('english'))
# Most common short words: (https://www3.nd.edu/~busiforc/handouts/cryptography/cryptography%20hints.html - University of Notre Dame) + some added based on our dataset
common_word_list = ['a', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 'back','be', 'been', 'boy', 'but', 'by', 'can','come', 'dad','day', 'did', 'do', 'for', 'food','from', 'get', 'go', 'god', 'good', 'got', 'has', 'have', 'he', 'her','hey', 'him', 'his', 'hope','how', 'in', 'is', 'it', 'its', 'know', 'let','love', 'man', 'me','mom', 'much', 'must', 'my', 'new', 'no', 'not', 'now', 'of','oh', 'old', 'on', 'one', 'or', 'our', 'out', 'put', 'say', 'see', 'she', 'so', 'some', 'son', 'that', 'the', 'they', 'this', 'to', 'too', 'two', 'up', 'us', 'use', 'want', 'was', 'way', 'we', 'who', 'will', 'with', 'yes', 'you', 'your']
# Create a list with stopwords and most common short words 
words_to_remove= list(stopwords_en)+common_word_list
# Convert words to upper case
words_to_remove_upp = ([word.upper() for word in words_to_remove])
# Capitalize each word (first letter in upper case)
words_to_remove_cap = ([word.capitalize() for word in words_to_remove])

# Create a list including lower case, upper case and capitalized words
words_to_remove_en=[]
words_to_remove_en= words_to_remove + words_to_remove_upp + words_to_remove_cap

# Sort list of words
words_to_remove_en.sort()
words_to_remove_upp.sort()

def abbreviation_metric(df,column,new_column):
    # Create empty list to add metric per row
    abbreviation = []
    # Loop to evaluate each row
    for value in df[column]:
        # Convert value to string
        value= str(value)  
        # Conditional for upper case text
        if str(value).isupper():
            # Remove upper case stopwords and short words 
            no_stopwords=remove_stopwords(value, stopwords=words_to_remove_upp)
            # Search for words with 2,3 capital letters followed by only one . and blank space
            abbreviations = re.findall(r'\b[A-Z]{2,3}\.(?!\.)[^\w.]*\s\b', no_stopwords)
        else:
            # Remove stopwords and short words
            no_stopwords= remove_stopwords(value, stopwords=words_to_remove_en)
            # Acronym pattern: Previous word is not upper case + 2-3 capital letters + next word is not upper case
            pattern_acronym= r'\b(?<![A-Z])([A-Z]{2,4})\s(?![A-Z])\b'
            #Tittles or date abbreviations: First letter in upper case + 1-2 lower case letters + only one .
            # (not other symbol) and blank space
            pattern_titles_days=r'\b[A-Z]{1}[a-z]{1,2}\.(?!\.)[^\w.]*\s\b'
            # Search for acronym, tittles or date abbreviations
            abbreviations = re.findall(pattern_acronym +'|'+ pattern_titles_days, no_stopwords)
        # Calculate the percentage of abbreviations in the text
        abbreviation_percentage= len(abbreviations)/len(str(value).split())
        # Normalize metric, taking the abbreviation percentage from the 100% and add the value into the list
        abbreviation.append(1-abbreviation_percentage)  
    # Create a new column including the abbreviation metric
    df[new_column] =abbreviation
    return df


  ## Metric 2: Spelling mistake metric
def spelling_mistake_metric(df,column,new_column):
    # Call SpellChecker function
    spell = SpellChecker()
    # Create empty list to add metric per row
    spelling_mistake_metric = []
    # Loop to evaluate each row
    for value in df[column]:
        # Remove tags, links, punctuation, numbers and contraction
        preprocessing_value = re.sub(r"@[A-Za-z0-9_]+|http\S+|www.\S+|'m|'d|n't|'s",'',str(value), flags=re.I)
        preprocessing_value = strip_punctuation(preprocessing_value)
        preprocessing_value = strip_numeric(preprocessing_value)
        # Tokenize the text into words
        words = preprocessing_value.split()
        # Correct words
        correction = [spell.correction(word) for word in words]
        # Comparison between word sets 
        different_words = list(set(words).symmetric_difference(set(correction)))
        # Calculate the percentage of spelling mistake
        spelling_mistake_percentage = round(len(different_words)/2) / len(str(words).split())
        # Normalize metric, taking the spelling mistake percentage from the 100% and add the value into the list
        spelling_mistake_metric.append(1-spelling_mistake_percentage)  
    # Create a new column including the unknown words metric
    df[new_column] = spelling_mistake_metric
    return df


## Metric 3: Unknown words metric
def unknown_word_metric(df,column,new_column):
    # Call SpellChecker function
    spell = SpellChecker()
    # Create empty list to add found unknown words
    unknown_words_metric = []
    # Loop to review each row
    for value in df[column]:
        # Remove tags, links, punctuation and numbers
        preprocessing_value = re.sub(r'@[A-Za-z0-9_]+|http+','',str(value), flags=re.I)
        preprocessing_value = strip_punctuation(preprocessing_value)
        preprocessing_value = strip_numeric(preprocessing_value)
        # Tokenize the text into words
        words = preprocessing_value.split()
        # Find unknown words
        unknown_words = spell.unknown(words)
        # Calculate the percentage of unknown words
        unknown_words_percentage = len(unknown_words) / len(str(value).split())
        # Normalize metric, taking the unknown words percentage from the 100% and add the value into the list
        unknown_words_metric.append(1-unknown_words_percentage)  
    # Create a new column including the unknown words metric
    df[new_column] = unknown_words_metric
    return df


## Metric 4: Grammatical sentence metric
def grammatical_sentence_metric(df, column,new_column):
    nlp = spacy.load("en_core_web_sm") 
    # Create empty list to add metric per row
    grammatical_sentences = []
    # Creation of patterns needed to identify a grammatical sentence
    noun_phrase = r'(PRON\s|PROPN\s|(DET\s((ADJ\s){0,3})?(NOUN|PROPN)\s))'
    verb_phrase = '((AUX\s)?VERB\s{1,2})'
    prepositional_phrase = '(ADP\s)'
    conjunction = '(CCONJ\s | SCONJ\s)'
    # Loop to evaluate each row
    for value in df[column]:
        # Tokenize the text into sentences
        value = nlp(str(value))
        sentences = list(value.sents)
        for sentence in sentences:
            sent_structure = [word.pos_ for word in sentence]
            sent_structure = ' '.join(sent_structure)
            # Create list with grammatical sentences
            grammatical_sents = re.findall(noun_phrase + verb_phrase + noun_phrase + "?" + prepositional_phrase 
                                           + "?.*" + conjunction + "?" + noun_phrase + "?" + verb_phrase + "?",
                                           sent_structure)
        # Calculate the percentage of grammatical sentences
        grammatical_percentage = len(grammatical_sents) / len(sentences)

        # No normalization needed, add the value into the list
        grammatical_sentences.append(grammatical_percentage)
    # Create a new column including grammatical sentence metric
    df[new_column] = grammatical_sentences



## Metric 5: Lexical density metric
def lexical_density_metric(df, column,new_column):
    nlp = spacy.load("en_core_web_sm")
    # Create empty list to add metric per row
    lexical_density = []
    # Loop to evaluate each row
    for value in df[column]:
        # Remove punctuation and symbols
        value = strip_punctuation(str(value))
        value = re.sub(r"â|€|™",'',str(value), flags=re.I)
        # Process text with spacy
        value = nlp(str(value))
        # Identify noun, adjetives, verbs and adverbs and add these words into a list
        lexical_density_list = [word.text for word in value if word.pos_ in ('NOUN', 'ADJ', 'VERB', 'ADV')]
        # Calculate percentage of content words (noun, adjetives, verbs and advers)
        lexical_density_percentage = len(lexical_density_list) / len(str(value).split())
        # No normalization needed, add the value into the list
        lexical_density.append(lexical_density_percentage)
    # Create a new column including the lexical density metric
    df[new_column] = lexical_density
    return df


## Metric 6: Lexical diversity metric
def lexical_diversity_metric(df, column,new_column):
    # Create empty list to add metric per row
    lexical_diversity=[]
    # Loop to evaluate each row
    for value in df[column]:
        # Calculate the lexical diversity: unique words / total amount of words
        lexical_diversity_percentage=(len(set(str(value).split())) / len(str(value).split()))
        # No need to be normalize 
        lexical_diversity.append(lexical_diversity_percentage)
    # Create a new column including the lexical diversity metric
    df[new_column] = lexical_diversity
    return df


## Metric 7: Stop words metric
def stop_word_metric(df,column,new_column):
    # Create empty list to add metric per row
    no_stopwords_list = []
    # Loop to evaluate each row
    for value in df[column]:
        # Convert value to string
        value= str(value)  
        # Remove stopwords (previously loaded for abbreviation metric)
        no_stopwords = remove_stopwords(value, stopwords=stopwords_en)
        # Calculate the percentage of characters without symbols and punctuation in the text
        no_stopword_percentage= len(no_stopwords.split())/len(str(value).split())
        # No normalization needed, so just add the value into the list 
        no_stopwords_list.append(no_stopword_percentage)  
    # Create a new column including the abbreviation metric
    df[new_column] = no_stopwords_list
    return df


## Metric 8: Average sentence length metric
def average_sentence_length_metric(df, column,new_column):
    nlp = spacy.load("en_core_web_sm")
    # Create empty list to add metric per row
    average_sentence_length=[]
    # Loop to evaluate each row
    for value in df[column]:
        value_nlp = nlp(str(value))
        sentences = list(value_nlp.sents)
        # Calculate average sentence length grade
        avg_sent_length_red=(len(str(value))-(len(str(value))/len(sentences)))/len(sentences)
        # If the result is higher than 100, make it 100 
        if avg_sent_length_red > 100:
            avg_sent_length_red=100
        # Normalize metric, reduce the average sentence length grade from the 100, divide the value into 100 
        # and add the value into the list
        average_sentence_length.append(1-(avg_sent_length_red/100))
    # Create a new column including the average sentence lengthd metric
    df[new_column] = average_sentence_length
    return df


## Metric 9: Uppercased word metric
def uppercased_word_metric(df, column,new_column):
    # Create empty list to add metric per row
    uppercased_words=[]
    # Loop to evaluate each row
    for value in df[column]:
        # Tokenize the text into words
        words= str(value).split()
        # Identify uppercase words
        uppercased_word_list = [word for word in words if word.isupper()]
        # Calculate uppercased words percentage
        uppercased_words_percentage=len(uppercased_word_list) / len(words)
        # Normalize metric, taking the uppercased words percentage from the 100% and add the value into the list
        uppercased_words.append(1-uppercased_words_percentage)
    # Create a new column including the uppercased word metric
    df[new_column] = uppercased_words
    return df


## Metric 10: Punctuation metric
def symbol_punctuation_metric(df,column,new_column):
    # Create empty list to add metric per row
    symbols_punctuation = []
    # Loop to evaluate each row
    for value in df[column]:
        value= str(value)
        # Remove punctuation
        preprocessing_value = strip_punctuation(value)
        # Remove other symbols
        preprocessing_value= re.sub(r'[¿¡’„”“’°]|…|£||#|═{2,}|⭗|▒|✓','',preprocessing_value, flags=re.I)
        # Remove spaces to only count characters, this apply for the value and preprocessing value
        value=re.sub(r'\s{1,}', '',value)
        preprocessing_value= re.sub(r'\s{1,}', '',preprocessing_value)
        # Calculate the percentage of characters without symbols and punctuation in the text
        symb_punct_percentage= len(preprocessing_value)/len(value)
        # No normalization needed, so just add the value into the list
        symbols_punctuation.append(symb_punct_percentage)  
    # Create a new column including the symbol and punctuation metric
    df[new_column] = symbols_punctuation
    return df

def column_average(df, columns, new_column):
    df[new_column] = df[columns].mean(axis=1)
    return df

