# Sistem za Automatsku Klasifikaciju Proizvoda

Ovaj projekat predstavlja rešenje za automatsku kategorizaciju proizvoda na osnovu njihovog naziva koristeći mašinsko učenje. Model je razvijen kao deo sistema za online trgovinu kako bi se ubrzao proces unosa novih artikala i smanjila mogućnost ljudske greške.

## Struktura Projekta

- `products.csv`: Skup podataka sa više od 30,000 proizvoda.
- `eda_and_model_development.py`: Skript za inicijalnu analizu podataka (EDA) i razvoj modela.
- `train_model.py`: Finalni Python skript za treniranje i čuvanje modela.
- `predict_category.py`: Interaktivni skript za testiranje modela (unos naziva, dobijanje kategorije).
- `model.pkl`: Trenirani model (Logistic Regression).
- `tfidf_vectorizer.pkl`: Sačuvani TF-IDF vektoraizer za transformaciju teksta.
- `README.md`: Dokumentacija projekta.

## Instalacija i Pokretanje

### 1. Instalacija zavisnosti
Za pokretanje projekta potrebno je imati instaliran Python 3 i sledeće biblioteke:
```bash
pip install pandas scikit-learn joblib
```

### 2. Treniranje modela
Da biste istrenirali model na priloženim podacima, pokrenite:
```bash
python3 train_model.py
```
Ovaj skript će očistiti podatke, normalizovati kategorije, istrenirati Logistic Regression model i sačuvati ga u `model.pkl`.

### 3. Interaktivno testiranje
Nakon što je model istreniran, možete ga testirati unosom sopstvenih naziva proizvoda:
```bash
python3 predict_category.py
```

## Rezultati Modela

Model postiže visoku preciznost na test skupu (oko 95%). Korišćena je Logistic Regression sa TF-IDF vektorizacijom, što se pokazalo kao robustno i efikasno rešenje za klasifikaciju teksta u ovom domenu.

## Primeri Testiranja
| Naziv proizvoda | Očekivana Kategorija | Rezultat Modela |
| :--- | :--- | :--- |
| iphone 7 32gb gold | Mobile Phones | Mobile Phones |
| olympus e m10 mark iii | Digital Cameras | Digital Cameras |
| bosch wap28390gb 8kg | Washing Machines | Washing Machines |
| smeg sbs8004po | Fridge Freezers | Fridge Freezers |
