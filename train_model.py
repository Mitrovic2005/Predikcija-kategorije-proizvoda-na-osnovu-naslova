import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def train_and_save_model(data_path='products.csv'):
    print("Učitavanje podataka...")
    df = pd.read_csv(data_path)
    
    # Čišćenje podataka
    df = df.dropna(subset=['Product Title', ' Category Label'])
    
    # Mapiranje sličnih kategorija (normalizacija)
    category_map = {
        'fridge': 'Fridges',
        'CPU': 'CPUs',
        'Mobile Phone': 'Mobile Phones'
    }
    df[' Category Label'] = df[' Category Label'].str.strip().replace(category_map)
    
    X = df['Product Title']
    y = df[' Category Label']
    
    # Podela na trening i test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Vektorizacija teksta...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    print("Treniranje modela (Logistic Regression)...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluacija
    y_pred = model.predict(X_test_tfidf)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Čuvanje
    joblib.dump(model, 'model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    print("\nModel i vektoraizer su uspešno sačuvani u 'model.pkl' i 'tfidf_vectorizer.pkl'.")

if __name__ == "__main__":
    train_and_save_model()
