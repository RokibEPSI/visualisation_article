import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
import itertools

#1 Charger le fichier TSV

""" le chargement  et aussi le complement d'une fonctionnalité """
file_path = 'marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv'
data = pd.read_csv(file_path, sep='\t')

# Sélection des colonnes pertinentes
columns_to_keep = ['Product Id', 'Product Price', 'Product Available Inventory', 'Product Reviews Count', 'Product Category']
clustering_data = data[columns_to_keep]

# Renommer les colonnes pour une meilleure lisibilité
clustering_data.rename(columns={
    'Product Id': 'Product_ID',
    'Product Price': 'Price',
    'Product Available Inventory': 'Inventory_Level',
    'Product Reviews Count': 'Quantity_Sold',
    'Product Category': 'Category'
}, inplace=True)

# Suppression des lignes avec des valeurs manquantes
clustering_data.dropna(inplace=True)

# Encodage de la colonne 'Category'
label_encoder = LabelEncoder()
clustering_data['Category_Encoded'] = label_encoder.fit_transform(clustering_data['Category'])



""" 2. Prétraitement des Données : 
o Seules les colonnes numériques (Price, Quantity_Sold, Inventory_Level) 
sont sélectionnées pour le clustering. 
o Les données sont normalisées avec MinMaxScaler pour garantir un 
clustering équitab"""


#2 Normalisation des colonnes numériques pour le clustering
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(clustering_data[['Price', 'Inventory_Level', 'Quantity_Sold', 'Category_Encoded']])

""" 3. Clustering : 
o L'algorithme k-Means est appliqué avec k=3 clusters pour regrouper les 
produits en trois catégories de performance : Élevé, Moyen, Faible.   """


# Appliquer k-Means avec 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clustering_data['Cluster'] = kmeans.fit_predict(normalized_data)

""" 4. Associer les Clusters à des Labels : 
o Les clusters sont mappés à des étiquees de performance (par ex., 
Faible, Moyen, Élevé).  """

# Ajouter des étiquettes aux clusters en fonction des moyennes
cluster_means = clustering_data.groupby('Cluster')[['Price', 'Inventory_Level', 'Quantity_Sold', 'Category_Encoded']].mean().sort_values('Quantity_Sold', ascending=False)
labels = {idx: label for idx, label in enumerate(['Élevé', 'Moyen', 'Faible'])}
clustering_data['Cluster_Label'] = clustering_data['Cluster'].map(labels)

""" 5. Visualisation : 
o Un diagramme de dispersion montre les clusters dans un espace 2D, 
meant en évidence la relation entre Price et Quantity_Sold.    


"""

# Visualisation des clusters
plt.figure(figsize=(10, 6))
for label, color in zip(['Élevé', 'Moyen', 'Faible'], ['red', 'blue', 'green']):
    cluster_subset = clustering_data[clustering_data['Cluster_Label'] == label]
    plt.scatter(cluster_subset['Price'], cluster_subset['Quantity_Sold'], label=label, alpha=0.6)

plt.title("Clusters des produits par performance")
plt.xlabel("Prix (normalisé)")
plt.ylabel("Quantité vendue (normalisée)")
plt.legend()
plt.show()

# Résumé des clusters
summary = clustering_data.groupby('Cluster_Label')[['Price', 'Inventory_Level', 'Quantity_Sold', 'Category_Encoded']].mean()
summary.columns = ['Prix Moyen', 'Niveau de Stock Moyen', 'Quantité Vendue Moyenne', 'Catégorie Moyenne']
print(summary)
# Votre code principal jusqu'à la visualisation initiale
# Charger et préparer les données (le reste de votre code est ici)
# ...

# Visualisation initiale des clusters (déjà dans votre code)
plt.figure(figsize=(10, 6))
for label, color in zip(['Élevé', 'Moyen', 'Faible'], ['red', 'blue', 'green']):
    cluster_subset = clustering_data[clustering_data['Cluster_Label'] == label]
    plt.scatter(cluster_subset['Price'], cluster_subset['Quantity_Sold'], label=label, alpha=0.6)

plt.title("Clusters des produits par performance")
plt.xlabel("Prix (normalisé)")
plt.ylabel("Quantité vendue (normalisée)")
plt.legend()
plt.show()

# Visualisation des clusters en fonction des paires de fonctionnalités
features = ['Price', 'Inventory_Level', 'Quantity_Sold', 'Category_Encoded']
pairs = list(itertools.combinations(features, 2))

plt.figure(figsize=(12, len(pairs) * 4))

for idx, (x_feature, y_feature) in enumerate(pairs, 1):
    plt.subplot(len(pairs), 1, idx)
    
    for label, color in zip(['Élevé', 'Moyen', 'Faible'], ['red', 'blue', 'green']):
        cluster_subset = clustering_data[clustering_data['Cluster_Label'] == label]
        plt.scatter(cluster_subset[x_feature], cluster_subset[y_feature], label=label, alpha=0.6)
    
    plt.title(f"Clusters : {x_feature} vs {y_feature}")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend()

plt.tight_layout()
plt.show()
