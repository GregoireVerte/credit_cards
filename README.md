# Predykcja Niespłacenia Kredytu przez Klientów Banku

## 1. Cel Projektu (Business Problem)

Celem tego projektu było zbudowanie i optymalizacja modelu uczenia maszynowego do predykcji, czy klient banku nie spłaci swojego zobowiązania kredytowego w następnym miesiącu. Poprawna identyfikacja klientów wysokiego ryzyka pozwala na minimalizację strat finansowych i podjęcie odpowiednich działań prewencyjnych. Główną metryką sukcesu był **F1-score**, z uwzględnieniem wysokiego **Recall** dla klasy pozytywnej (niespłacenie kredytu - klasa 1 --> default).

---

## 2. O Zbiorze Danych

Zbiór danych pochodzi z zadań kursu ML Kodilla.

**Opis zmiennych:**

This research employed a binary variable, default payment next month (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
* `X1`: Amount of the given credit (NT dollar).: it includes both the individual consumer credit and his/her family (supplementary) credit.
* `X2`: Gender (1 = male; 2 = female).
* `X3`: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
* `X4`: Marital status (1 = married; 2 = single; 3 = others).
* `X5`: Age (year).
* `X6` - `X11`: History of past payment (from April to September, 2005). 
    * The measurement scale is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; ...; 9 = payment delay for nine months and above.
* `X12`-`X17`: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005.
* `X18`-`X23`: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.
* `Y`: **(Zmienna docelowa)** Default payment next month (1 = Yes, 0 = No).

---

## 3. Zastosowana Metodologia i Przebieg Prac

Projekt został zrealizowany w sposób kompleksowy, obejmując następujące etapy:
1.  **Wstępna Analiza Danych (EDA):** Zrozumienie rozkładu zmiennych i ich korelacji wzajemnych oraz ze zmienną docelową.
2.  **Czyszczenie i Przygotowanie Danych:** Ujednolicenie nazw kolumn, typów danych i obsługa błędów w danych.
3.  **Zaawansowana Inżynieria Cech:** Stworzenie kilkudziesięciu nowych, potężnych cech na podstawie cech numerycznych, kategorycznych oraz ich interakcji (np. `PAY_TO_BILL_RATIO`, `RISK_WEIGHTED_UNPAID`, `LIMIT_X_MAX_STATUS`), które poprawiły jakość modeli.
4.  **Budowa i Walidacja Pipeline'u:** Stworzenie w pełni odtwarzalnego pipeline'u w `scikit-learn` i `imblearn`, uwzględniającego preprocessing, inżynierię cech, selekcję cech i modelowanie.
5.  **Testowanie i Strojenie Modeli:** Przeprowadzenie wyczerpujących testów (`GridSearchCV` i `RandomizedSearchCV`) dla szerokiej gamy algorytmów, w tym:
    * Modele Liniowe (`Logistic Regression`, `SGDClassifier`)
    * Modele Oparte na Odległości (`KNN`)
    * Maszyny Wektorów Nośnych (`SVC` z różnymi jądrami)
    * Modele Drzewiaste i Ensemblowe (`Decision Tree`, `Random Forest`, `AdaBoost`, `Gradient Boosting`, `HistGradientBoosting`, `XGBoost`, `LightGBM`, `CatBoost`)
    * Zaawansowane techniki do walki z niezbalansowanymi danymi (`BalancedRandomForestClassifier`, `BalancedBaggingClassifier`)
6.  **Budowa Modeli Finałowych:** Stworzenie ostatecznych, bardzo silnych modeli ensemblowych (`VotingClassifier` i `StackingClassifier`).
7.  **Optymalizacja Progu Decyzyjnego:** Znalezienie optymalnego progu klasyfikacji w celu maksymalizacji metryk biznesowych.
8.  **Interpretacja Wyników:** Użycie biblioteki **SHAP** do zrozumienia, które cechy miały największy wpływ na decyzje najlepszego modelu.

---

## 4. Ostateczne Wyniki i Wnioski

Po przeprowadzeniu kilkudziesięciu eksperymentów (śledzonych za pomocą **MLflow**), zidentyfikowano grupę najlepszych modeli, które osiągnęły "sufit" wydajności dla tego zbioru danych.

**Najlepszy model:** **CatBoost (MVS)**
* **F1-score:** **0.541**
* **Recall:** 0.625
* **Precision:** 0.476
* **AUC:** 0.778

**Kluczowe wnioski:**
* Największy wzrost jakości modelu przyniosła **zaawansowana inżynieria cech**, a nie wybór konkretnego algorytmu.
* Wszystkie najlepsze modele (CatBoost, VotingClassifier, LightGBM, Random Forest) osiągnęły bardzo zbliżone wyniki, co świadczy o osiągnięciu maksymalnego pułapu predykcyjnego dla dostępnych danych.
* Analiza SHAP dla najlepszego modelu wykazała, że kluczowe dla predykcji były cechy opisujące historię płatności (np. `MAX_PAY_STATUS`), cechy interakcji (np. 'PAY_TO_BILL_RATIO_APR') oraz cechy numeryczne agregujące (np. `MAX_PAY_AMT`).
* Optymalizacja progu decyzyjnego pozwoliła na dalsze, niewielkie, ale istotne z biznesowego punktu widzenia, poprawienie wyników `F1-score` i `Recall`.

---

## 5. Jak Uruchomić Projekt

1.  Sklonuj repozytorium.
2.  Stwórz środowisko i zainstaluj wymagane biblioteki: `pip install -r requirements.txt`
3.  Uruchom notatniki w kolejności:
    * `credit_cards.ipynb`
    * `credit_cards_2_modeling.ipynb`