import pandas as pd
import streamlit as st
import google.generativeai as genai

# Récupération sécurisée depuis les secrets Streamlit
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

st.title("📚 Mon Recommandeur de Livres")

# Chargement de la base Notion
@st.cache_data
def load_data():
    df = pd.read_csv("livres.csv")
    return df

df_livres = load_data()

# Optionnel : Afficher un aperçu de la liste
if st.checkbox("Voir la liste des livres disponibles"):
    st.dataframe(df_livres)

# Interface utilisateur
user_query = st.text_input("Quel genre de livre recherchez-vous ou quelle est votre humeur actuelle ?")

if st.button("Obtenir des recommandations"):
    if user_query:
        with st.spinner("L'IA analyse la bibliothèque..."):
            # Conversion du CSV en texte pour donner du contexte à l'IA
            contexte_livres = df_livres.to_string(index=False)
            
            prompt = f"""
            Tu es un bibliothécaire expert. Voici la liste des livres disponibles dans ma bibliothèque :
            {contexte_livres}
            
            En te basant uniquement ou prioritairement sur cette liste, recommande 3 livres qui correspondent à la demande suivante : "{user_query}".
            Pour chaque livre, donne le titre, l'auteur et explique brièvement pourquoi il correspond à la demande.
            """
            
            response = model.generate_content(prompt)
            st.markdown(response.text)
    else:
        st.warning("Veuillez entrer une demande.")