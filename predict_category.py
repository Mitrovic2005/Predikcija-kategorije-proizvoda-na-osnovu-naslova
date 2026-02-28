import joblib
import sys
import os

def predict_category():
    # Putanje do modela i vektoraizera
    model_path = 'model.pkl'
    tfidf_path = 'tfidf_vectorizer.pkl'
    
    # Provera da li model postoji
    if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
        print("Greška: Model ili vektoraizer nisu pronađeni. Prvo pokrenite 'train_model.py'.")
        return
    
    # Učitavanje modela i vektoraizera
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    
    print("\n--- Sistem za automatsku klasifikaciju proizvoda ---")
    print("Unesite naziv proizvoda (ili 'exit' za izlaz):")
    
    while True:
        try:
            product_title = input("\nNaziv proizvoda: ")
            if product_title.lower() == 'exit':
                print("Izlaz iz programa.")
                break
            
            if not product_title.strip():
                print("Molimo unesite validan naziv proizvoda.")
                continue
                
            # Transformacija ulaza
            X_input = tfidf.transform([product_title])
            
            # Predviđanje
            prediction = model.predict(X_input)[0]
            
            # Verovatnoća
            probabilities = model.predict_proba(X_input)
            max_prob = probabilities.max()
            
            print(f"Predložena kategorija: {prediction} (pouzdanost: {max_prob:.2%})")
            
        except KeyboardInterrupt:
            print("\nPrekid programa.")
            break
        except Exception as e:
            print(f"Došlo je do greške: {e}")

if __name__ == "__main__":
    predict_category()
