import streamlit as st
import torch
import nltk
import numpy as np
import re
from nltk.corpus import stopwords
import torch.nn as nn

# Download necessary NLTK data
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

# Function to preprocess input text
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [re.sub(r'\W+', '', token) for token in tokens if re.sub(r'\W+', '', token)]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return tokens

# Load pre-trained model
def load_model(model_path, vocab_size, embedding_dim, hidden_dim):
    model = LanguageModel(vocab_size, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Generate text
def generate_text(model, start_text, max_length, word2index, index2word):
    model.eval()
    words = preprocess_text(start_text)
    state_h, state_c = model.init_state(batch_size=1)

    for _ in range(max_length):
        x = torch.tensor([[word2index.get(w, word2index['']) for w in words]], dtype=torch.long)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(index2word[word_index])

    return ' '.join(words)

# Load vocabulary and other resources
with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = preprocess_text(text)
vocab = list(set(tokens))
word2index = {word: i for i, word in enumerate(vocab)}
index2word = {i: word for i, word in enumerate(vocab)}

# Streamlit app UI
st.title("Text Generation with LSTM")
st.write("Generate text using a pre-trained LSTM model based on 'The Adventures of Sherlock Holmes'.")

# Hyperparameters
embedding_dim = 50
hidden_dim = 100
vocab_size = len(vocab)
model_path = 'model.pth'

# Load the model
model = load_model(model_path, vocab_size, embedding_dim, hidden_dim)

# Text input from user
start_text = st.text_input("Enter the start text for text generation", "Sherlock Holmes")
max_length = st.slider("Select the maximum length of the generated text", min_value=10, max_value=100, value=50)

if st.button("Generate Text"):
    with st.spinner('Generating text...'):
        generated_text = generate_text(model, start_text, max_length, word2index, index2word)
    st.subheader("Generated Text:")
    st.write(generated_text)
