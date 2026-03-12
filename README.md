# Kaggle — House Prices: Advanced Regression Techniques

Prédiction du prix de vente de maisons à Ames, Iowa.  
Compétition Kaggle : [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## Résultat

Meilleure soumission : **submission_v6_tuned.csv**  
Métrique locale (RMSE CV 5-fold) : **0.1234**

---

## Approche

**Modèle final : Stacking Ensemble (5 modèles + méta-modèle)**

| Modèle | Rôle |
|--------|------|
| Ridge (α=15) | Base |
| Lasso (α=0.0005) | Base |
| ElasticNet (α=0.001) | Base |
| GradientBoosting (300 est., lr=0.05, subsample=0.8) | Base |
| XGBoost (600 est., lr=0.05) | Base |
| Ridge (α=10) | Méta-modèle |

**Prétraitement :**
- Gestion des valeurs manquantes par type (None / 0 / mode / médiane par quartier)
- 17 features créées : `TotalSF`, `TotalBathrooms`, `HouseAge`, `QualityTimesArea`, etc.
- 276 features totales après one-hot encoding
- Normalisation : `RobustScaler`
- Transformation cible : `log1p(SalePrice)` → `expm1` pour les prédictions

---

## Structure

```
kaggle-house-prices/
├── house_prices_kaggle.ipynb   # Pipeline complet
├── submission_v6_tuned.csv     # Fichier de soumission
├── .gitignore
└── README.md
```

> Les fichiers `train.csv`, `test.csv`, `sample_submission.csv` sont à télécharger depuis [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) et placer dans le même dossier.

---

## Installation

```bash
pip install numpy pandas scikit-learn xgboost lightgbm jupyter
jupyter notebook house_prices_kaggle.ipynb
```
