# Feature Extraction Module

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import re
import string
from collections import Counter

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

class FakeNewsFeatureExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.emotion_lexicon = self._load_emotion_lexicon()
    
    def _load_emotion_lexicon(self):
        nrc_df = pd.read_csv("NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", sep='\t', names=['word','emotion','association'], encoding="utf-8")
        nrc_df = nrc_df[nrc_df['association']==1]
        emotion_lexicon = nrc_df.groupby('emotion')['word'].apply(set).to_dict()
        return emotion_lexicon
    
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", '', text)
        return text
    
    def extract_pos_features(self, text):
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        pos_counts = Counter([tag for word, tag in pos_tags])
        total_tokens = len(tokens)
        if total_tokens == 0:
            return [0] * 8
        
        features = []
        first_person = pos_counts.get('PRP', 0)
        total_pronouns = first_person + pos_counts.get('PRP$', 0)
        features.append(first_person / total_tokens)
        modal_verbs = pos_counts.get('MD', 0)
        features.append(modal_verbs / total_tokens)
        adjectives = pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + pos_counts.get('JJS', 0)
        features.append(adjectives / total_tokens)
        nouns = pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)
        verbs = pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)
        noun_verb_ratio = nouns / verbs if verbs > 0 else nouns
        features.append(noun_verb_ratio)
        adverbs = pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + pos_counts.get('RBS', 0)
        features.append(adverbs / total_tokens)
        past_verbs = pos_counts.get('VBD', 0) + pos_counts.get('VBN', 0)
        present_verbs = pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)
        past_present_ratio = past_verbs / present_verbs if present_verbs > 0 else past_verbs
        features.append(past_present_ratio)
        conjunctions = pos_counts.get('CC', 0)
        features.append(conjunctions / total_tokens)
        determiners = pos_counts.get('DT', 0)
        features.append(determiners / total_tokens)
        return features
    
    def extract_structural_features(self, text):
        if not text:
            return [0] * 6
        features = []
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        features.append(avg_word_length)
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        features.append(avg_sentence_length)
        features.append(len(words))
        punctuation_count = len([char for char in text if char in string.punctuation])
        punctuation_density = punctuation_count / len(text) if text else 0
        features.append(punctuation_density)
        capital_count = len([char for char in text if char.isupper()])
        capital_ratio = capital_count / len(text) if text else 0
        features.append(capital_ratio)
        paragraph_count = len(text.split('\n\n')) if '\n\n' in text else 1
        features.append(paragraph_count)
        return features
    
    def extract_emotional_features(self, text):
        if not text:
            return [0] * 10
        words = word_tokenize(text.lower())
        # Remove stopwords for emotional analysis (it shouldn't be deleted for structural part without realizing)
        words = [w for w in words if w not in self.stop_words]
        features = []
        emotions = ['joy', 'anger', 'fear', 'sadness', 'trust', 'surprise', 'anticipation', 'disgust', 'positive', 'negative']
        for emotion in emotions:
            emotion_words = self.emotion_lexicon.get(emotion, set())
            emotion_count = sum(1 for word in words if word in emotion_words)
            emotion_density = emotion_count / len(words) if words else 0
            features.append(emotion_density)
        return features
    
    def extract_all_features(self, text):
        processed_text = self.preprocess_text(text)
        pos_features = self.extract_pos_features(processed_text)
        structural_features = self.extract_structural_features(processed_text)
        emotional_features = self.extract_emotional_features(processed_text)
        return pos_features + structural_features + emotional_features
    
    def get_feature_names(self):
        pos_names = [
            'first_person_pronoun_density',
            'modal_verb_density', 
            'adjective_density',
            'noun_verb_ratio',
            'adverb_density',
            'past_present_verb_ratio',
            'conjunction_density',
            'determiner_density'
        ]
        structural_names = [
            'avg_word_length',
            'avg_sentence_length', 
            'total_word_count',
            'punctuation_density',
            'capital_letter_ratio',
            'paragraph_count'
        ]
        emotional_names = [
            'joy_density',
            'anger_density',
            'fear_density', 
            'sadness_density',
            'trust_density',
            'surprise_density',
            'anticipation_density',
            'disgust_density',
            'positive_sentiment_density',
            'negative_sentiment_density'
        ]
        return pos_names + structural_names + emotional_names

def extract_features_from_dataframe(df, text_column='text'):
    extractor = FakeNewsFeatureExtractor()
    features_list = []
    for text in df[text_column]:
        features = extractor.extract_all_features(text)
        features_list.append(features)
    feature_names = extractor.get_feature_names()
    features_df = pd.DataFrame(features_list, columns=feature_names)
    
    features_df['text'] = df[text_column].values
    features_df['label'] = df['label'].values

    cols = ['text', 'label'] + [col for col in features_df.columns if col not in ['text', 'label']]
    features_df = features_df[cols]
    
    return features_df
