"""
Projet Machine Learning : Supervisé & Non Supervisé
Dataset : Heart Disease (UCI / Kaggle)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, ConfusionMatrixDisplay,
                              precision_score, recall_score, f1_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
np.random.seed(42)
n = 918

n0, n1 = 411, 507
age0 = np.random.normal(50, 9, n0).clip(28, 77).astype(int)
age1 = np.random.normal(57, 8, n1).clip(28, 77).astype(int)
sex0 = np.random.choice(['M', 'F'], n0, p=[0.68, 0.32])
sex1 = np.random.choice(['M', 'F'], n1, p=[0.87, 0.13])
chol0 = np.random.normal(210, 52, n0).clip(85, 564).astype(int)
chol1 = np.random.normal(190, 50, n1).clip(85, 564).astype(int)
maxhr0 = np.random.normal(149, 22, n0).clip(60, 202).astype(int)
maxhr1 = np.random.normal(124, 24, n1).clip(60, 202).astype(int)
oldpeak0 = np.abs(np.random.normal(0.4, 0.7, n0)).round(1)
oldpeak1 = np.abs(np.random.normal(1.4, 1.2, n1)).round(1)
rbp0 = np.random.normal(128, 16, n0).clip(92, 200).astype(int)
rbp1 = np.random.normal(135, 19, n1).clip(92, 200).astype(int)

def rc(n, p): return np.random.choice(range(len(p)), n, p=p)

df0 = pd.DataFrame({
    'Age': age0, 'Sex': sex0,
    'ChestPainType': rc(n0, [0.25, 0.35, 0.28, 0.12]),
    'RestingBP': rbp0, 'Cholesterol': chol0,
    'FastingBS': rc(n0, [0.88, 0.12]),
    'RestingECG': rc(n0, [0.55, 0.22, 0.23]),
    'MaxHR': maxhr0,
    'ExerciseAngina': rc(n0, [0.83, 0.17]),
    'Oldpeak': oldpeak0,
    'ST_Slope': rc(n0, [0.05, 0.30, 0.65]),
    'CA': rc(n0, [0.72, 0.18, 0.07, 0.02, 0.01]),
    'Thal': rc(n0, [0.02, 0.06, 0.75, 0.17]),
    'HeartDisease': 0
})
df1 = pd.DataFrame({
    'Age': age1, 'Sex': sex1,
    'ChestPainType': rc(n1, [0.75, 0.12, 0.09, 0.04]),
    'RestingBP': rbp1, 'Cholesterol': chol1,
    'FastingBS': rc(n1, [0.68, 0.32]),
    'RestingECG': rc(n1, [0.65, 0.17, 0.18]),
    'MaxHR': maxhr1,
    'ExerciseAngina': rc(n1, [0.52, 0.48]),
    'Oldpeak': oldpeak1,
    'ST_Slope': rc(n1, [0.35, 0.37, 0.28]),
    'CA': rc(n1, [0.44, 0.28, 0.17, 0.09, 0.02]),
    'Thal': rc(n1, [0.02, 0.10, 0.43, 0.45]),
    'HeartDisease': 1
})
df = pd.concat([df0, df1], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Ajout de valeurs nulles et doublons (réalisme)
null_idx = np.random.choice(df.index, 15, replace=False)
df.loc[null_idx[:8],  'Cholesterol'] = np.nan
df.loc[null_idx[8:],  'RestingBP']   = np.nan
df = pd.concat([df, df.iloc[:5]], ignore_index=True)

# ─────────────────────────────────────────────
print("=== APERÇU DU DATASET ===")
print(df.head())
print(f"\nDimensions        : {df.shape}")
print(f"Valeurs nulles    :\n{df.isnull().sum()}")
print(f"Doublons          : {df.duplicated().sum()}")
print(f"Variable cible    :\n{df['HeartDisease'].value_counts()}")

# ─────────────────────────────────────────────
# 2. PRÉPARATION DES DONNÉES
# ─────────────────────────────────────────────
df = df.drop_duplicates()
df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].median())
df['RestingBP']   = df['RestingBP'].fillna(df['RestingBP'].median())

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

print(f"\n=== APRÈS NETTOYAGE ===")
print(f"Dimensions : {df.shape} | Nulls : {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# FIGURE 1 — Distribution + Corrélation
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Exploration des données – Heart Disease Dataset", fontsize=14, fontweight='bold')

counts = df['HeartDisease'].value_counts().sort_index()
axes[0].bar(['Sain (0)', 'Malade (1)'], counts.values,
            color=['#2196F3', '#F44336'], edgecolor='white', linewidth=1.5)
axes[0].set_title('Distribution de la variable cible', fontweight='bold')
axes[0].set_ylabel('Nombre de patients')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

corr = df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'HeartDisease']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlBu_r',
            linewidths=0.5, ax=axes[1])
axes[1].set_title('Matrice de corrélation', fontweight='bold')
plt.tight_layout()
plt.show()   # <-- fenêtre 1 : prends ton screen ici

# ─────────────────────────────────────────────
# 3. APPRENTISSAGE SUPERVISÉ
# ─────────────────────────────────────────────
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc  = sc.transform(X_test)

clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
clf.fit(X_train_sc, y_train)
y_pred = clf.predict(X_test_sc)

acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)

print(f"\n=== RÉSULTATS CLASSIFICATION ===")
print(f"Accuracy : {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=['Sain', 'Malade']))

# ─────────────────────────────────────────────
# FIGURE 2 — Matrice de confusion + Importance
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Résultats – Random Forest  |  Accuracy = {acc:.2%}",
             fontsize=13, fontweight='bold')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sain', 'Malade'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Matrice de confusion', fontweight='bold')

imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values()
axes[1].barh(imp.index, imp.values,
             color=plt.cm.viridis(np.linspace(0.2, 0.85, len(imp))))
axes[1].set_title("Importance des variables (Random Forest)", fontweight='bold')
axes[1].axvline(imp.mean(), color='red', ls='--', alpha=0.7, label='Moyenne')
axes[1].legend()
plt.tight_layout()
plt.show()   # <-- fenêtre 2 : prends ton screen ici

# ─────────────────────────────────────────────
# 4. APPRENTISSAGE NON SUPERVISÉ — K-MEANS
# ─────────────────────────────────────────────
X_clust = sc.fit_transform(X)

# Méthode du coude
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_clust)
    inertias.append(km.inertia_)

# ─────────────────────────────────────────────
# FIGURE 3 — Méthode du coude
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 11), inertias, 'o-', color='#1976D2', lw=2.5, ms=8)
ax.fill_between(range(1, 11), inertias, alpha=0.1, color='#1976D2')
ax.axvline(x=3, color='#F44336', ls='--', lw=2, label='k optimal = 3')
ax.set_xlabel('Nombre de clusters (k)', fontsize=12)
ax.set_ylabel('Inertie (WCSS)', fontsize=12)
ax.set_title('Méthode du Coude – Choix du nombre de clusters',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()   # <-- fenêtre 3 : prends ton screen ici

# K-Means k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_clust)
df['Cluster'] = clusters

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_clust)

print(f"\n=== CLUSTERING K-MEANS (k=3) ===")
print(df['Cluster'].value_counts())
print(f"Variance expliquée PCA : {pca.explained_variance_ratio_.sum():.2%}")

# ─────────────────────────────────────────────
# FIGURE 4 — Visualisation clusters PCA
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Clustering K-Means (k=3) — Visualisation PCA",
             fontsize=13, fontweight='bold')

cols_k = ['#1976D2', '#43A047', '#E53935']
for i, (c, l) in enumerate(zip(cols_k, ['Cluster 0', 'Cluster 1', 'Cluster 2'])):
    m = clusters == i
    axes[0].scatter(X_pca[m, 0], X_pca[m, 1], c=c, label=l,
                    alpha=0.55, s=30, edgecolors='none')
cp2 = pca.transform(kmeans.cluster_centers_)
axes[0].scatter(cp2[:, 0], cp2[:, 1], c='black', marker='X',
                s=200, zorder=5, label='Centroïdes')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0].set_title('Distribution des clusters')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.2)

profile = df.groupby('Cluster')[['Age', 'MaxHR', 'Oldpeak', 'Cholesterol']].mean()
profile.T.plot(kind='bar', ax=axes[1], color=cols_k, edgecolor='white', width=0.7)
axes[1].set_title('Profil moyen par cluster')
axes[1].set_ylabel('Valeur moyenne')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend(['Cluster 0', 'Cluster 1', 'Cluster 2'], fontsize=9)
axes[1].grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.show()   # <-- fenêtre 4 : prends ton screen ici

# ─────────────────────────────────────────────
# FIGURE 5 — Comparaison supervisé / non supervisé
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Analyse Comparative : Supervisé vs Non Supervisé",
             fontsize=13, fontweight='bold')

ct = pd.crosstab(df['Cluster'], df['HeartDisease'], normalize='index') * 100
ct.plot(kind='bar', ax=axes[0], color=['#2196F3', '#F44336'],
        edgecolor='white', width=0.65)
axes[0].set_title('Proportion HeartDisease par cluster', fontweight='bold')
axes[0].set_ylabel('%')
axes[0].set_xticklabels([f'Cluster {i}' for i in range(3)], rotation=0)
axes[0].legend(['Sain (0)', 'Malade (1)'])
axes[0].grid(True, alpha=0.2, axis='y')

metrics = ['Accuracy', 'Precision\n(Malade)', 'Recall\n(Malade)', 'F1\n(Malade)']
vals = [
    acc,
    precision_score(y_test, y_pred, pos_label=1),
    recall_score(y_test, y_pred, pos_label=1),
    f1_score(y_test, y_pred, pos_label=1)
]
bars = axes[1].bar(metrics, vals,
                   color=['#3F51B5', '#4CAF50', '#FF9800', '#E91E63'],
                   edgecolor='white', width=0.5)
axes[1].set_ylim(0, 1.1)
axes[1].set_title('Métriques supervisé (Random Forest)', fontweight='bold')
axes[1].set_ylabel('Score')
for b, v in zip(bars, vals):
    axes[1].text(b.get_x() + b.get_width() / 2, v + 0.01,
                 f'{v:.2%}', ha='center', fontsize=10, fontweight='bold')
axes[1].grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.show()   # <-- fenêtre 5 : prends ton screen ici

print("\n✅ Terminé !")
