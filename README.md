# NLP_st125496_A2

# Text Generation with Language Model

This project is a simple implementation of a text generation model built using PyTorch, NLTK, and Streamlit. The model is trained on the Project Gutenberg text of *The Adventures of Sherlock Holmes* and can generate text based on a user-provided prompt through a Streamlit web interface.

## Features
- **Text Preprocessing:** Tokenizes, removes stop words, and numericalizes the input text.
- **Language Model:** Uses an LSTM-based architecture to predict the next word in a sequence.
- **Streamlit Interface:** Allows users to input a prompt and generate text interactively.
- **Model Training:** Includes an easy-to-follow script for training the language model.

---

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Download the dataset (already included in the script):
   - The dataset is sourced from Project Gutenberg and is automatically downloaded during execution.

4. Install NLTK data:
   Run the following script to ensure all necessary NLTK components are downloaded:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

---

## How to Run
### Training the Model
1. Run the script to train the model:
   ```bash
   python train_model.py
   ```
2. The trained model will be saved as `model.pth` in the project directory.

### Running the Streamlit App
1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open the provided localhost link in your web browser.
3. Enter a text prompt and generate text interactively.

---

## File Structure
- **`train_model.py`**: Script to preprocess the text, train the LSTM model, and save the trained model.
- **`app.py`**: Streamlit application for interactive text generation.
- **`dataset.txt`**: The raw text data extracted from Project Gutenberg (*The Adventures of Sherlock Holmes*).
- **`model.pth`**: The saved model state dictionary after training.
- **`requirements.txt`**: List of required Python libraries for the project.

---

## Text Generation Workflow
1. **Data Preprocessing**
   - The raw text is tokenized, converted to lowercase, and stripped of punctuation and stop words.
   - Sequences of 5 words are created for training.

2. **Model Architecture**
   - Embedding Layer: Converts words into dense vectors.
   - LSTM Layers: Two-layer LSTM for sequence modeling.
   - Fully Connected Layer: Maps the hidden state to vocabulary size for prediction.

3. **Training**
   - Cross-entropy loss is used for optimization.
   - The model is trained in batches of 32 for 10 epochs.

4. **Text Generation**
   - Starts with a user-provided prompt.
   - Generates text word by word using the trained model.

---

## Requirements
- Python 3.8+
- Required libraries (install via `pip install -r requirements.txt`):
  - `torch`
  - `nltk`
  - `numpy`
  - `streamlit`

---

## Usage Example
1. Enter a prompt like "Sherlock Holmes solves the".
2. The model generates a continuation such as:
   ```
   Sherlock Holmes solves the case with precision and logic unmatched by any detective of his era.
   ```

---

## Notes
- The generated text quality depends on the model's training and the dataset used.
- You can experiment with different datasets and hyperparameters to improve performance.

---

## Future Improvements
- Add support for larger datasets.
- Enhance the model architecture for better text generation quality.
- Include a user-configurable interface for setting generation parameters.

---

## License
This project is licensed under the MIT License. Feel free to use and modify it as needed.

---

## Acknowledgements
- Project Gutenberg for providing the dataset.
- PyTorch and NLTK for their powerful libraries.
- Streamlit for the easy-to-use web application framework.

