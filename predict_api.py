import pandas as pd
import numpy as np
import pickle
from scipy.sparse import load_npz
import json
from flask import Flask, request, jsonify

# Inicializa a aplicação Flask
app = Flask(__name__)

# --- Carrega os Componentes do Modelo ---
MODEL_DIR = './saved_model'  # Assume que os arquivos do modelo estão no mesmo diretório

def load_model():
    """Carrega os componentes do modelo."""
    try:
        # Carrega a matriz de similaridade
        similarity_matrix = load_npz(f'{MODEL_DIR}/similarity_matrix.npz')
        # Carrega o mapeamento user_id para índice
        with open(f'{MODEL_DIR}/user_id_to_index.pkl', 'rb') as f:
            user_id_to_index = pickle.load(f)
        # Carrega o histórico de treinamento
        with open(f'{MODEL_DIR}/train_history.pkl', 'rb') as f:
            train_history = pickle.load(f)
        # Carrega o recomendador de tendências
        with open(f'{MODEL_DIR}/trending_recommender.pkl', 'rb') as f:
            trending_recommender = pickle.load(f)
        f.close() # Fecha o arquivo
        return True, similarity_matrix, user_id_to_index, train_history, trending_recommender
    except FileNotFoundError:
        print("Erro: Arquivos do modelo não encontrados. Certifique-se de que eles estão no diretório especificado.")
        return False, None, None, None, None
    except Exception as e:
        print(f"Um erro inesperado ocorreu durante o carregamento do modelo: {e}")
        return False, None, None, None, None

def recommend_top_n(similarity_matrix, user_index, top_n=10):
    """Recomenda os top-N itens para um determinado usuário com base na similaridade de cosseno."""
    user_similarities = similarity_matrix[user_index]  # Obtém as similaridades do usuário
    # Obtém os índices dos usuários mais similares (excluindo o próprio usuário)
    similar_user_indices = np.argsort(user_similarities.toarray()[0])[::-1][1:]
    return similar_user_indices[:top_n]  # Retorna os top-N usuários mais similares

def generate_recommendations(user_id, top_n=10):
    """
    Gera recomendações para um determinado usuário (existente ou novo).

    Args:
        user_id: O ID do usuário.
        top_n: Número de recomendações a serem geradas.

    Returns:
        Uma lista de IDs de itens recomendados, ou None se ocorrer um erro.
    """
    model_loaded, similarity_matrix, user_id_to_index, train_history, trending_recommender = load_model()

    if not model_loaded:
        return None  # Ou lança uma exceção

    # --- Tratamento de Cold Start ---
    if user_id not in user_id_to_index:
        # Cold start: Usa o recomendador de tendências
        max_timestamp_val = max(trending_recommender.keys())
        recommendations = trending_recommender.get(max_timestamp_val, [])[:top_n]

    # --- Warm Start (Usuário Existente) ---
    else:
        user_index = user_id_to_index[user_id]  # Obtém o índice do usuário
        similar_users_indices = recommend_top_n(similarity_matrix, user_index, top_n)  # Obtém os usuários similares

        # Mapeia os índices de volta para user_ids, *então* obtém o histórico.
        final_recommendations = []
        # Mapeamento reverso: índice para user_id
        index_to_user_id = {index: user_id for user_id, index in user_id_to_index.items()}
        for similar_user_index in similar_users_indices:
            similar_user_id = index_to_user_id[similar_user_index]  # Obtém o user_id do usuário similar
            # Obtém o índice do histórico do usuário (a partir das chaves de user_id_to_index)
            user_history_index = list(user_id_to_index.keys()).index(similar_user_id)
            final_recommendations.extend(train_history[user_history_index])  # Adiciona o histórico do usuário

        # Remove duplicatas e limita ao top_n
        recommendations = list(dict.fromkeys(final_recommendations))[:top_n]

    return recommendations

@app.route('/recommend', methods=['GET'])
def recommend():
    """Endpoint da API para recomendações."""
    user_id = request.args.get('user_id')

    if not user_id:
        return jsonify({'error': 'Parâmetro user_id ausente'}), 400  # Requisição inválida

    recommendations = generate_recommendations(user_id)

    if recommendations is None:
        return jsonify({'error': 'Modelo não carregado ou outro erro do servidor'}), 500  # Erro interno do servidor

    return jsonify({'user_id': user_id, 'recommendations': recommendations}), 200  # OK

# if __name__ == '__main__':
#     app.run(debug=False, port=5000)  # <-- Desative o modo de depuração para testes de memória.