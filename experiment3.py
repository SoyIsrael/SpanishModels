import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset

# Step 1: Connect to SQLite and load data
database_path = './utils/data/wikibooks.sqlite'
conn = sqlite3.connect(database_path)

query_sp = "SELECT title, body_text, abstract FROM es LIMIT 800;"  # Reduced Spanish data
query_en = "SELECT title, body_text, abstract FROM en LIMIT 200;"  # English data
df_spanish = pd.read_sql_query(query_sp, conn)
df_english = pd.read_sql_query(query_en, conn)

# Add language column
df_spanish['language'] = 'spanish'
df_english['language'] = 'english'

# Combine data
df_combined = pd.concat([df_spanish[['title', 'body_text', 'language']], df_english[['title', 'body_text', 'language']]], ignore_index=True)
df_combined['label'] = df_combined['language'].apply(lambda x: 1 if x == 'english' else 0)  # 1 for English, 0 for Spanish

# Clean the body_text column
df_combined['body_text'] = df_combined['body_text'].str.replace(r'\W', ' ', regex=True).str.lower()

# Split data into train and test
X = df_combined['body_text']
y = df_combined['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Tokenize the text using a multilingual tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('bert-base-multilingual-cased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=60)

# Convert to Hugging Face Dataset
train_data = pd.DataFrame({'text': X_train, 'label': y_train})
test_data = pd.DataFrame({'text': X_test, 'label': y_test})

train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Tokenize the data
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Step 3: Fine-tune a multilingual DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=0.3,             # Reduced number of epochs
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True,                        # Mixed precision (optional for faster training if on GPU)
    eval_steps=500,                   # Evaluate every 500 steps
    evaluation_strategy="steps",      # Enables evaluation during training
    load_best_model_at_end=True,      # Loads the best model at the end of training
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]  # Stop after 1 epoch of no improvement
)

# Step 5: Train the model
trainer.train()

# Step 6: Evaluate the model
predictions = trainer.predict(test_dataset)

# Step 7: Display classification report
print("Classification Report:")
print(classification_report(y_test, predictions.predictions.argmax(axis=1)))
