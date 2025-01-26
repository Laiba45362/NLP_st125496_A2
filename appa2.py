import streamlit as st
import requests
import nltk
import re
import numpy as np
import torch
import torch.nn as nn
import os
from nltk.corpus import stopwords

# Ensure necessary NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')

# Define the LanguageModel class
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, prev_state):
        x = self.embedding(x)
        x, state = self.lstm(x, prev_state)
        x = self.fc(x)
        return x, state

    def init_state(self, batch_size=1):
        return (torch.zeros(2, batch_size, self.lstm.hidden_size),
                torch.zeros(2, batch_size, self.lstm.hidden_size))

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Lowercasing
    tokens = [token.lower() for token in tokens]

    # Removing punctuation and special characters
    tokens = [re.sub(r'\W+', '', token) for token in tokens if re.sub(r'\W+', '', token)]

    # Removing stop words (optional)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Add a special token for unknown words
    tokens.append('<UNK>')

    # Numericalization
    vocab = list(set(tokens))
    word2index = {word: i for i, word in enumerate(vocab)}
    index2word = {i: word for i, word in enumerate(vocab)}

    # Creating sequences
    sequence_length = 5
    sequences = []
    for i in range(len(tokens) - sequence_length):
        sequences.append(tokens[i:i + sequence_length])

    # Convert sequences to numerical indices
    input_sequences = []
    for sequence in sequences:
        input_sequences.append([word2index[word] for word in sequence])

    # Convert to numpy array
    input_sequences = np.array(input_sequences)

    return input_sequences, vocab, word2index, index2word

def generate_text(model, start_text, word2index, index2word, max_length=50):
    model.eval()
    words = start_text.split()
    state_h, state_c = model.init_state(batch_size=1)

    for _ in range(max_length):
        x = torch.tensor([[word2index.get(w, word2index['<UNK>']) for w in words]], dtype=torch.long)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(index2word[word_index])

    return ' '.join(words)

# Load the dataset and preprocess it
dataset_path = 'dataset.txt'
input_sequences, vocab, word2index, index2word = load_and_preprocess_data(dataset_path)

# Load the trained model
model_path = 'model.pth'
model = LanguageModel(len(vocab), embedding_dim=50, hidden_dim=100)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Streamlit app
st.title("Text Generation with Pre-trained LSTM")

start_text = st.text_input("Enter the start text for text generation", "harry potter is")
if st.button("Generate Text"):
    with st.spinner('Generating text...'):
        generated_text = generate_text(model, start_text, word2index, index2word, max_length=50)
    st.write("Generated Text:")
    st.write(generated_text)
