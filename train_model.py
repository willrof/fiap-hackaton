import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, csr_matrix
import pickle
from scipy.sparse import vstack, save_npz
import gc #Garbage collector
from tqdm import tqdm

# Definições das Funções (incluídas aqui para organização)

def create_features(train_df, item_df):
    """
    Cria features para o modelo de recomendação, APENAS para os dados de treinamento.

    Args:
        train_df: O DataFrame de treinamento limpo.
        item_df: O DataFrame de itens limpo.

    Returns:
        Um DataFrame com as features criadas, pronto para o modelo.
    """

    # 1. Cria a feature is_logged_in (se o usuário está logado)
    train_df.loc[:, 'is_logged_in'] = (train_df['userType'] == 'Logged').astype(int)

    # 2. Adiciona a coluna 'dataset' (para diferenciar treino/validação depois, se necessário)
    train_df.loc[:, 'dataset'] = 'train'  # Usa .loc para evitar warnings

    # 3. Calcula o timestamp MÁXIMO por usuário (crucial para features baseadas em tempo)
    max_timestamp_per_user = train_df.groupby('userId')['timestampHistory'].apply(
        lambda x: np.nanmax([ts for arr in x for ts in arr]) if any(arr.size > 0 for arr in x) else np.nan
    ).reset_index()
    max_timestamp_per_user.columns = ['userId', 'max_timestamp']

    # 4. Expande os dados de treinamento para ter uma linha por interação
    # Isso facilita o cálculo da idade do artigo e outras features
    exploded_train = []
    for _, row in train_df.iterrows():
        for i in range(len(row['history'])):
            exploded_train.append({
                'userId': row['userId'],
                'page': row['history'][i],
                'timestamp': row['timestampHistory'][i],
                'timeOnPage': row['timeOnPageHistory'][i],
                'scrollPercentage': row['scrollPercentageHistory'][i],
                'is_logged_in': row['is_logged_in'],
                'dataset': 'train'
            })
    exploded_train_df = pd.DataFrame(exploded_train)

    # 5. Combina os dados expandidos (neste caso, só temos treino)
    exploded_combined_df = exploded_train_df

    # 6. Junta com os dados dos itens para obter o timestamp 'modified'
    exploded_combined_df = exploded_combined_df.merge(item_df[['page', 'modified']], on='page', how='left')

    # 7. Calcula article_age_hours (idade do artigo em horas)
    exploded_combined_df['article_age_hours'] = (exploded_combined_df['timestamp'] - exploded_combined_df['modified'].astype(int) // 10**6) / (60 * 60 * 1000)

    # 8. Calcula features médias por usuário (tempo na página, porcentagem de rolagem)
    user_avg_features = exploded_combined_df.groupby('userId').agg({
        'timeOnPage': 'mean',
        'scrollPercentage': 'mean'
    }).reset_index()
    user_avg_features.columns = ['userId', 'user_average_time_on_page', 'user_average_scroll_percentage']

    # 9. Junta as features por usuário de volta ao DataFrame combinado
    exploded_combined_df = exploded_combined_df.merge(user_avg_features, on='userId', how='left')

    # 10. Junta o max_timestamp por usuário
    exploded_combined_df = exploded_combined_df.merge(max_timestamp_per_user, on='userId', how='left')

    # 11. Calcula time_since_last_interaction (tempo desde a última interação)
    exploded_combined_df['time_since_last_interaction'] = exploded_combined_df['max_timestamp'] - exploded_combined_df['timestamp']

    # 12. Separa de volta em treino
    final_train_df = exploded_combined_df[exploded_combined_df['dataset'] == 'train'].drop(columns=['dataset'])

    # 13. Ajustes finais (remove a coluna 'modified')
    final_train_df = final_train_df.drop(columns=['modified'])

    return final_train_df

def create_bow_embeddings(train_df):
    """
    Cria embeddings Bag-of-Words (BOW) para os históricos dos usuários

    Args:
        train_df: O DataFrame de treinamento com as features.

    Returns:
        train_bow: Uma matriz esparsa representando os embeddings BOW para o conjunto de treinamento.
        vectorizer: O objeto CountVectorizer ajustado.
    """

    # Combina 'userId' e 'page' em uma única string para o CountVectorizer
    train_df['user_history_str'] = train_df['userId']

    # Inicializa o CountVectorizer
    vectorizer = CountVectorizer()

    # Ajusta nos dados de *treinamento* e transforma os dados de treinamento
    train_bow = vectorizer.fit_transform(train_df['user_history_str'])

    return train_bow, vectorizer

def combine_features(bow_matrix, df):
    """
    Combina os embeddings BOW com outras features numéricas.

    Args:
        bow_matrix: A matriz BOW esparsa.
        df: O DataFrame com as features (treinamento ou validação).

    Returns:
        Uma matriz esparsa com todas as features combinadas.
    """

    # Extrai as features numéricas
    numerical_features = df[[
        'is_logged_in', 'article_age_hours',
        'user_average_time_on_page', 'user_average_scroll_percentage',
        'time_since_last_interaction'
    ]].values  # Converte para um array NumPy

    # Converte as features numéricas para uma matriz esparsa (importante para hstack)
    numerical_features_sparse = csr_matrix(numerical_features)

    # Empilha horizontalmente (hstack) as features BOW e numéricas
    combined_features = hstack([bow_matrix, numerical_features_sparse])

    return combined_features

def build_trending_recommender(train_df_cleaned, item_df_cleaned, time_window_hours=24):
    """
    Constrói um recomendador de itens em alta com base em uma janela de tempo.

    Args:
        train_df_cleaned: O DataFrame de treinamento limpo.
        item_df_cleaned: O DataFrame de itens limpo.
        time_window_hours: O tamanho da janela de tempo em horas.

    Returns:
        Um dicionário onde as chaves são timestamps (dentro da janela de tempo)
        e os valores são listas de IDs de itens em alta (páginas), ordenados por
        uma pontuação combinada de cliques e porcentagem de rolagem.
    """

    trending_recommender = {}

    # Calcula o tempo de início da janela de tempo
    max_timestamp = train_df_cleaned['timestampHistory'].apply(lambda x: np.nanmax(x)).max()
    time_window_start = max_timestamp - (time_window_hours * 60 * 60 * 1000)  # Converte horas para milissegundos

    # Filtra as interações dentro da janela de tempo
    df_in_window = train_df_cleaned[train_df_cleaned['timestampHistory'].apply(lambda x: np.nanmax(x) >= time_window_start)]

    # Cria uma lista para armazenar os dados de interação (página, timestamp, cliques, rolagem)
    interactions = []
    rows_skipped = 0  # Contagem de linhas ignoradas
    for _, row in df_in_window.iterrows():
        history = row['history']
        timestamps = row['timestampHistory']
        clicks = row['numberOfClicksHistory']
        scrolls = row['scrollPercentageHistory']

        # --- VERIFICAÇÃO DE COMPRIMENTO ROBUSTA ---
        if not (len(history) == len(timestamps) == len(clicks) == len(scrolls)):
            rows_skipped += 1
            continue  # Ignora a linha inteira se os comprimentos não coincidirem

        for i in range(len(history)):
            interactions.append({
                'page': history[i],
                'timestamp': timestamps[i],
                'clicks': clicks[i],
                'scroll': scrolls[i]
            })

    print(f"Número de linhas ignoradas devido a comprimentos inconsistentes: {rows_skipped}")

    # Converte para DataFrame
    interactions_df = pd.DataFrame(interactions)
    # Filtra interações anteriores a time_window_start
    interactions_df = interactions_df[interactions_df['timestamp'] >= time_window_start]

    # Junta com os dados dos itens para obter o timestamp 'issued' para o cálculo da idade
    interactions_df = interactions_df.merge(item_df_cleaned[['page', 'issued']], on='page', how='left')
    # Calcula a idade do artigo em horas
    interactions_df['article_age_hours'] = (interactions_df['timestamp'] - interactions_df['issued'].astype(int) // 10**6) / (60 * 60 * 1000)

    # Agrega as interações: soma cliques e média da porcentagem de rolagem
    aggregated_interactions = interactions_df.groupby('page').agg({
        'clicks': 'sum',
        'scroll': 'mean',
        'timestamp': 'max',  # Mantém o timestamp da interação mais recente
        'article_age_hours': 'mean'  # Mantém a média da idade do artigo
    }).reset_index()

    # Normaliza as contagens de cliques e porcentagens de rolagem
    aggregated_interactions['clicks_normalized'] = (aggregated_interactions['clicks'] - aggregated_interactions['clicks'].min()) / (aggregated_interactions['clicks'].max() - aggregated_interactions['clicks'].min())
    aggregated_interactions['scroll_normalized'] = (aggregated_interactions['scroll'] - aggregated_interactions['scroll'].min()) / (aggregated_interactions['scroll'].max() - aggregated_interactions['scroll'].min())

    # Cria uma pontuação combinada (favorecendo a porcentagem de rolagem)
    aggregated_interactions['trending_score'] = (0.3 * aggregated_interactions['clicks_normalized']) + (0.7 * aggregated_interactions['scroll_normalized'])

    # Ordena por pontuação de tendência (decrescente) e timestamp (decrescente)
    sorted_items = aggregated_interactions.sort_values(by=['trending_score', 'timestamp'], ascending=[False, False])

    # Cria o dicionário de recomendação com timestamps como chaves
    for timestamp in sorted(interactions_df['timestamp'].unique()):
        # Filtra os itens relevantes para o timestamp atual
        relevant_items = sorted_items[sorted_items['timestamp'] <= timestamp]
        trending_recommender[timestamp] = relevant_items['page'].tolist()

    return trending_recommender

def train_model(train_parquet_path, item_parquet_path, sample_size=10000, output_dir='.', chunk_size=1000):
    """
    Treina o modelo de recomendação e salva os componentes necessários.
    """

    # Carrega os DataFrames limpos
    train_df_cleaned = pd.read_parquet(train_parquet_path)
    item_df_cleaned = pd.read_parquet(item_parquet_path)

    # Amostra os usuários (ANTES da engenharia de features, mas aplicado DEPOIS)
    sampled_users = train_df_cleaned['userId'].drop_duplicates().sample(n=sample_size, random_state=42)

    # Engenharia de Features (nos dados COMPLETOS)
    final_train_df = create_features(train_df_cleaned, item_df_cleaned)

    # Filtra para os usuários amostrados APÓS a engenharia de features
    final_train_df_sampled = final_train_df[final_train_df['userId'].isin(sampled_users)].copy()
    print(f"Shape do final_train_df_sampled: {final_train_df_sampled.shape}")

    # 2. Cria os Embeddings BOW
    train_bow, vectorizer = create_bow_embeddings(final_train_df_sampled)

    # 3. Combina as Features
    train_features_combined = combine_features(train_bow, final_train_df_sampled)

    # --- Cálculo Iterativo da Similaridade de Cosseno (Esparsa) ---
    num_users = train_features_combined.shape[0]  # Correto: número de linhas após a amostragem
    similarity_matrix_sparse = None  # Inicializa como None

    for i in tqdm(range(0, num_users, chunk_size), desc="Calculando Similaridade"):
        # Obtém um pedaço da matriz de features
        chunk = train_features_combined[i:min(i + chunk_size, num_users)]

        # Calcula a similaridade para este pedaço (saída esparsa).  Deixa como float64
        similarity_chunk = cosine_similarity(chunk, train_features_combined, dense_output=False)

        # Garante que os índices sejam int32
        similarity_chunk.indices = similarity_chunk.indices.astype(np.int32)
        similarity_chunk.indptr = similarity_chunk.indptr.astype(np.int32)

        # Empilha verticalmente (constrói eficientemente uma matriz esparsa grande)
        if similarity_matrix_sparse is None:
            similarity_matrix_sparse = similarity_chunk
        else:
            gc.collect()
            # CRÍTICO: Mantém os índices como int32 ao empilhar.
            similarity_matrix_sparse = vstack([similarity_matrix_sparse, similarity_chunk], format='csr')
            similarity_matrix_sparse.indices = similarity_matrix_sparse.indices.astype(np.int32)
            similarity_matrix_sparse.indptr = similarity_matrix_sparse.indptr.astype(np.int32)

        # Libera a memória do pedaço
        del similarity_chunk
        gc.collect()

    # 5. Cria train_history (lista de listas) - MUITO MAIS RÁPIDO
    train_history = final_train_df_sampled.groupby('userId')['page'].apply(list).tolist()

    # 6. Cria o mapeamento user_id_to_index. Usa os usuários AMOSTRADOS
    user_id_to_index = {user_id: index for index, user_id in enumerate(final_train_df_sampled['userId'].unique())}

    # 7. Constrói o recomendador de tendências
    trending_recommender = build_trending_recommender(train_df_cleaned, item_df_cleaned, time_window_hours=24)

    # 8. Salva os componentes do modelo
    save_npz(f'{output_dir}/similarity_matrix.npz', similarity_matrix_sparse)  # Salva como matriz esparsa
    save_npz(f'{output_dir}/train_features_combined.npz', train_features_combined) # Salva train_features
    with open(f'{output_dir}/user_id_to_index.pkl', 'wb') as f:
        pickle.dump(user_id_to_index, f)
    with open(f'{output_dir}/train_history.pkl', 'wb') as f:
        pickle.dump(train_history, f)
    with open(f'{output_dir}/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(f'{output_dir}/trending_recommender.pkl', 'wb') as f:
        pickle.dump(trending_recommender, f)

    print("Treinamento do modelo completo. Componentes salvos.")

if __name__ == '__main__':
 # Realiza o treinamento
 train_model('train_df_cleaned.parquet', 'item_df_cleaned.parquet', sample_size=2000, output_dir='.', chunk_size=500)