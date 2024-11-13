import sqlite3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Connect to the SQLite database
database_path = './utils/data/wikibooks.sqlite'
conn = sqlite3.connect(database_path)

# Step 2: Define the SQL query to select the desired table (e.g., Spanish table)
query_sp = "SELECT title, body_text, abstract FROM es LIMIT 1000;"  # Assuming 'es' is the Spanish table
query_en = "SELECT title, body_text, abstract FROM en LIMIT 1000;"  # Assuming 'en' is the English table

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

# Step 6: Model Training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Step 8: Analyze features that contributed to the model
# Get the feature names
feature_names = vectorizer.get_feature_names_out()

# Get the coefficients of the model
coefficients = model.coef_[0]

# Create a DataFrame to view the features and their corresponding coefficients
features_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})

# Sort the DataFrame by coefficient value in descending order
features_df = features_df.sort_values(by='coefficient', ascending=False)

# Display the top features
print("Top positive features for predicting English:")
print(features_df.head(10))  # Top 10 features that contribute to predicting 'english'

print("\nTop negative features for predicting English:")
print(features_df.tail(10))  # Top 10 features that contribute to predicting 'spanish'
# Close the connection
conn.close()

# Optional: Display the first few rows of the English DataFrame
print(df_english['title'])
