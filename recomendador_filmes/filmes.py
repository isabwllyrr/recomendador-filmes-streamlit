import pandas as pd
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# 1. FUNÇÃO PARA CARREGAR DADOS
def carregar_dados(caminho_arquivo):
    try:
        dados = pd.read_csv(caminho_arquivo)
        st.success("Dados carregados com sucesso!")
        return dados
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

# Função para carregar dados de filmes e avaliações
def carregar_dados_movies_e_ratings(caminho_arquivo_movies, caminho_arquivo_ratings):
    movies = carregar_dados(caminho_arquivo_movies)
    ratings = carregar_dados(caminho_arquivo_ratings)
    return movies, ratings

# Função para preparar os dados para KNN
def preparar_dados(movies, ratings):
    media_avaliacoes = ratings.groupby('movieId')['rating'].mean().reset_index()
    dados_preparados = pd.merge(movies, media_avaliacoes, on='movieId', how='left')

    if 'rating' not in dados_preparados.columns:
        st.error("A coluna 'rating' não foi criada corretamente.")
        return None

    generos_encoded = dados_preparados['genres'].str.get_dummies(sep='|')
    dados_preparados = pd.concat([dados_preparados[['movieId', 'rating']], generos_encoded], axis=1)
    dados_preparados = dados_preparados.fillna(0)

    return dados_preparados

def recomendar_filmes_knn(movies, ratings, filme_titulo, k=5):
    filme_titulo = filme_titulo.strip().lower()
    dados_preparados = preparar_dados(movies, ratings)

    if dados_preparados is None:
        st.error("Não foi possível preparar os dados para KNN.")
        return

    filmes_encontrados = movies[movies['title'].str.lower().str.contains(filme_titulo)]

    if filmes_encontrados.empty:
        st.warning("Filme não encontrado. Tente novamente.")
        return

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(dados_preparados.drop(columns=['movieId']))

    for index, filme in filmes_encontrados.iterrows():
        try:
            filme_id = filme['movieId']
            if filme_id not in dados_preparados['movieId'].values:
                st.warning(f"Filme com ID {filme_id} não encontrado nos dados preparados.")
                continue

            indice_film_idx = dados_preparados[dados_preparados['movieId'] == filme_id].index[0]
            distances, indices = knn.kneighbors(dados_preparados.iloc[indice_film_idx].drop('movieId').values.reshape(1, -1))

            st.subheader(f"Filmes recomendados para '{filme['title']}':")
            for i in range(1, k):  # Começando de 1 para ignorar o próprio filme
                indice_recomendado = indices[0][i]
                filme_recomendado_id = dados_preparados.iloc[indice_recomendado]['movieId']
                filme_recomendado = movies[movies['movieId'] == filme_recomendado_id].iloc[0]
                st.write(f"Título: {filme_recomendado['title']} | Gêneros: {filme_recomendado['genres']}")
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")

# Função para recomendar filmes por gênero
def recomendar_filmes_por_genero(dados, genero):
    if dados is not None:
        filmes_filtrados = dados[dados['genres'].str.contains(genero, case=False, na=False)]
        if filmes_filtrados.empty:
            st.warning(f"Nenhum filme encontrado para o gênero: {genero}")
        else:
            st.subheader(f"Filmes recomendados para o gênero '{genero}':")
            for _, linha in filmes_filtrados.iterrows():
                st.write(f"Título: {linha['title']}, Gêneros: {linha['genres']}")
    else:
        st.error("Dados não carregados corretamente.")

# Função para exibir os gêneros
def exibir_generos(dados):
    if dados is not None:
        todos_os_generos = set()
        for generos in dados['genres']:
            todos_os_generos.update(generos.split('|'))
        st.subheader("Gêneros disponíveis:")
        for genero in sorted(todos_os_generos):
            st.write(f"- {genero}")
    else:
        st.error("Dados não carregados corretamente.")

# Função principal do Streamlit
def main():
    st.title("Recomendador de Filmes")

    # Carregar os dados
    caminho_arquivo_movies = st.file_uploader("Carregue o arquivo de filmes (movies.csv)", type="csv")
    caminho_arquivo_ratings = st.file_uploader("Carregue o arquivo de avaliações (ratings.csv)", type="csv")

    if caminho_arquivo_movies and caminho_arquivo_ratings:
        movies, ratings = carregar_dados_movies_e_ratings(caminho_arquivo_movies, caminho_arquivo_ratings)

        if movies is not None and ratings is not None:
            # Recomendações por Gênero
            genero = st.text_input("Digite o gênero que você está procurando:")
            if st.button("Recomendar Filmes por Gênero"):
                if genero:
                    recomendar_filmes_por_genero(movies, genero)
                else:
                    st.warning("Por favor, digite um gênero.")

            # Recomendações KNN
            filme_titulo = st.text_input("Digite o título do filme para recomendações:")
            if st.button("Recomendar Filmes com KNN"):
                if filme_titulo:
                    recomendar_filmes_knn(movies, ratings, filme_titulo)
                else:
                    st.warning("Por favor, digite um título de filme.")

            # Exibir gêneros
            if st.button("Exibir Gêneros Disponíveis"):
                exibir_generos(movies)

if __name__ == "__main__":
    main()
