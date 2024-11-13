import sqlite3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Connect to the SQLite database
database_path = './utils/data/wikibooks.sqlite'
conn = sqlite3.connect(database_path)

# Step 2: Define the SQL query to select the desired table (e.g., Spanish table)
# Reduce Spanish data to 200 rows for this hypothetical scenario
query_sp = "SELECT title, body_text, abstract FROM es LIMIT 200;"  # 200 Spanish samples
query_en = "SELECT title, body_text, abstract FROM en LIMIT 1000;"  # 1000 English samples

# Step 3: Load the data into a Pandas DataFrame
df_spanish = pd.read_sql_query(query_sp, conn)
df_english = pd.read_sql_query(query_en, conn)

# Add language column
df_spanish['language'] = 'spanish'
df_english['language'] = 'english'

# Combine the two DataFrames
df_combined = pd.concat([df_spanish[['title', 'body_text', 'language']], df_english[['title', 'body_text', 'language']]], ignore_index=True)

# Create labels (language will be used as the label here)
df_combined['label'] = df_combined['language']

# Clean the body_text column
df_combined['body_text'] = df_combined['body_text'].str.replace(r'\W', ' ', regex=True).str.lower()

# Prepare features and labels
X = df_combined['body_text']
y = df_combined['label']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Text Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- Experiment 1: Compare three models ---

# Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)
print("Logistic Regression classification report:")
print(classification_report(y_test, lr_pred))

# Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)
print("Naive Bayes classification report:")
print(classification_report(y_test, nb_pred))

# SVM Model
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
svm_pred = svm_model.predict(X_test_tfidf)
print("SVM classification report:")
print(classification_report(y_test, svm_pred))

# Step 8: Close the connection
conn.close()
