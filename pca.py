import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# load dataset into Pandas DataFrame
emb_file_path = 'sanfrancisco/sanfrancisco_raw_feature_crossing.embeddings'
pca_emb_file_path = 'sanfrancisco/sanfrancisco_pca_crossing_8d.embeddings'
df = pd.read_csv(emb_file_path, header=None, sep=' ', index_col=0)

rows_size, cols_size = df.shape
x = df

# Standardizing the features
x = StandardScaler().fit_transform(x)


def save_embeddings(embeddings, output_file_path):
    with open(output_file_path, 'w+') as f:
        for embedding in embeddings:
            f.write(' '.join(map(str, embedding)))
            f.write('\n')


def Pca():
    pca = PCA(n_components=8)
    print('training pca model: ')
    pca_transform = pca.fit_transform(x)
    save_embeddings(pca_transform, pca_emb_file_path)

    print('training done! and the embeddings saved!')


Pca()
