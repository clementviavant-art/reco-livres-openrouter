
import os
import re
import random
import pandas as pd
import streamlit as st
from openai import OpenAI

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Conseiller spécial pour la famille Viavant",
    page_icon="📚",
    layout="wide",
)

MODEL = "nvidia/nemotron-3.5-super-vl:free"
CSV_FILE = "livres.csv"

MAIN_TOKENS = 700
DISCOVERY_TOKENS = 250
HOME_SAMPLE_SIZE = 30

# ============================================================
# CLIENT OPENROUTER
# ============================================================

def get_client():
    """Crée le client OpenAI compatible avec OpenRouter."""
    api_key = None

    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        st.error(
            "Clé API OpenRouter absente. "
            "Ajoute OPENROUTER_API_KEY dans les Secrets Streamlit."
        )
        st.stop()

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=180.0,
        max_retries=1,
    )


client = get_client()

# ============================================================
# CHARGEMENT DU CATALOGUE
# ============================================================

@st.cache_data
def load_catalogue():
    """Charge le CSV et attribue un identifiant à chaque livre."""
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(
            f"Le fichier {CSV_FILE} est introuvable. "
            "Place-le dans le même dossier que app.py."
        )

    df = pd.read_csv(
        CSV_FILE,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
    ).fillna("")

    # Nettoyage des noms de colonnes
    df.columns = [col.strip() for col in df.columns]

    required = {"Auteur", "Titre du livre"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            "Le CSV doit contenir les colonnes "
            "'Auteur' et 'Titre du livre', séparées par des points-virgules."
        )

    df["Auteur"] = df["Auteur"].str.strip()
    df["Titre du livre"] = df["Titre du livre"].str.strip()

    df = df[
        (df["Auteur"] != "")
        & (df["Titre du livre"] != "")
    ].copy()

    df = df.drop_duplicates(
        subset=["Auteur", "Titre du livre"],
        keep="first",
    ).reset_index(drop=True)

    # Identifiants courts : le modèle renvoie les IDs,
    # et non les titres complets.
    df.insert(0, "ID", range(1, len(df) + 1))

    return df


try:
    catalogue = load_catalogue()
except Exception as exc:
    st.error(f"Impossible de charger le catalogue : {exc}")
    st.stop()


# ============================================================
# OUTILS
# ============================================================

def normalize_text(text):
    """Normalise une chaîne pour comparer les titres."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def catalogue_for_prompt(df):
    """
    Format compact transmis à l'IA.
    Le catalogue entier est conservé pour les recommandations principales.
    """
    lines = []

    for _, row in df.iterrows():
        lines.append(
            f"{int(row['ID'])}|{row['Auteur']}|{row['Titre du livre']}"
        )

    return "\n".join(lines)


def call_ai(system_prompt, user_prompt, max_tokens):
    """Appel unique à OpenRouter avec gestion des erreurs."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.4,
        extra_body={
            "reasoning": {
                "effort": "low",
                "exclude": True,
            }
        },
    )

    content = response.choices[0].message.content or ""
    usage = response.usage

    usage_info = None
    if usage:
        usage_info = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    return content.strip(), usage_info


def parse_recommendations(raw_text, df):
    """
    Format attendu :
    ID|identifiant|raison
    ID|identifiant|raison
    ID|identifiant|raison
    EXT|titre|auteur|raison

    Vérifie que les trois recommandations du catalogue
    correspondent réellement à des livres du CSV.
    """
    by_id = {
        str(int(row["ID"])): row
        for _, row in df.iterrows()
    }

    recommendations = []
    external = None

    for original_line in raw_text.splitlines():
        line = original_line.strip()

        if not line:
            continue

        # Retire les éventuelles puces ou clôtures Markdown.
        line = re.sub(r"^[-*•\s]+", "", line)
        line = line.replace("`", "")

        parts = [part.strip() for part in line.split("|")]

        if len(parts) >= 3 and parts[0].upper() == "ID":
            book_id = parts[1]

            if book_id not in by_id:
                continue

            row = by_id[book_id]

            if any(
                int(item["ID"]) == int(book_id)
                for item in recommendations
            ):
                continue

            recommendations.append({
                "ID": int(book_id),
                "titre": row["Titre du livre"],
                "auteur": row["Auteur"],
                "raison": "|".join(parts[2:]).strip(),
                "externe": False,
            })

        elif len(parts) >= 4 and parts[0].upper() == "EXT":
            title = parts[1]
            author = parts[2]
            reason = "|".join(parts[3:]).strip()

            catalogue_titles = {
                normalize_text(title_value)
                for title_value in df["Titre du livre"].tolist()
            }

            selected_titles = {
                normalize_text(item["titre"])
                for item in recommendations
            }

            if (
                title
                and author
                and normalize_text(title) not in catalogue_titles
                and normalize_text(title) not in selected_titles
            ):
                external = {
                    "titre": title,
                    "auteur": author,
                    "raison": reason,
                    "externe": True,
                }

    return recommendations[:3], external


# ============================================================
# SUGGESTIONS IA DE L'ACCUEIL
# ============================================================

def generate_home_suggestions(df):
    """
    Génère deux suggestions à partir d'un échantillon du catalogue.
    Le résultat est conservé dans la session Streamlit.
    """
    sample_size = min(HOME_SAMPLE_SIZE, len(df))
    sample = df.sample(n=sample_size, random_state=None)

    lines = [
        f"{int(row['ID'])}|{row['Auteur']}|{row['Titre du livre']}"
        for _, row in sample.iterrows()
    ]

    system_prompt = """
Tu es un conseiller littéraire chaleureux et pertinent.
Tu proposes des idées de lecture à une famille.
Réponds uniquement avec deux lignes au format :
ID|identifiant|raison courte

Choisis uniquement des identifiants présents dans la liste.
Choisis deux livres différents.
N'ajoute aucun texte avant ou après les deux lignes.
"""

    user_prompt = (
        "Voici une sélection de livres du catalogue :\n\n"
        + "\n".join(lines)
        + "\n\nChoisis deux livres variés qui donnent envie de lire."
    )

    raw_text, usage = call_ai(
        system_prompt,
        user_prompt,
        DISCOVERY_TOKENS,
    )

    selected, _ = parse_recommendations(raw_text, sample)

    if len(selected) < 2:
        raise ValueError(
            "L'IA n'a pas renvoyé deux suggestions exploitables. "
            "Clique sur le bouton pour réessayer."
        )

    return selected[:2], usage


# ============================================================
# RECOMMANDATIONS PERSONNALISÉES
# ============================================================

def recommend_books(
    df,
    genres,
    mood,
    length,
    audience,
    exclusions,
    extra_details,
):
    """
    Envoie l'intégralité du catalogue à l'IA.
    Une seule requête est utilisée pour les recommandations principales.
    """
    full_catalogue = catalogue_for_prompt(df)

    system_prompt = """
Tu es un excellent libraire et conseiller littéraire.
Tu dois sélectionner des livres pertinents en respectant strictement
les préférences du lecteur.

RÈGLES IMPORTANTES :
- La liste fournie est le catalogue complet disponible.
- Choisis exactement trois livres distincts dans ce catalogue.
- Utilise uniquement leurs identifiants numériques pour les désigner.
- Ajoute obligatoirement une quatrième recommandation extérieure
  au catalogue, qui ne figure pas dans la liste.
- La recommandation extérieure doit être un vrai livre identifiable,
  avec un titre et un auteur.
- Ne répète aucun livre.
- Respecte les exclusions indiquées.
- Les raisons doivent être personnalisées et courtes.
- Ne fournis ni introduction, ni conclusion, ni Markdown.
- N'invente jamais un identifiant de livre du catalogue.

FORMAT DE SORTIE STRICT :
ID|identifiant|raison courte
ID|identifiant|raison courte
ID|identifiant|raison courte
EXT|titre du livre|auteur|raison courte

Remplace les champs par les vraies valeurs.
Renvoie exactement quatre lignes.
"""

    preferences = f"""
PRÉFÉRENCES DU LECTEUR

Genres recherchés : {", ".join(genres) if genres else "Libre choix"}
Ambiance souhaitée : {mood or "Libre choix"}
Longueur souhaitée : {length}
Public : {audience}
Livres, genres ou éléments à éviter : {exclusions or "Aucune exclusion"}
Précisions supplémentaires : {extra_details or "Aucune"}

CATALOGUE COMPLET
Format : ID|Auteur|Titre du livre

{full_catalogue}

Sélectionne les livres les plus pertinents parmi l'intégralité de cette liste.
La quatrième recommandation doit être extérieure au catalogue.
"""

    raw_text, usage = call_ai(
        system_prompt,
        preferences,
        MAIN_TOKENS,
    )

    recommendations, external = parse_recommendations(
        raw_text,
        df,
    )

    return recommendations, external, usage, raw_text


# ============================================================
# INTERFACE
# ============================================================

st.title("📚 Conseiller spécial pour la famille Viavant")

st.write(
    "Trouve ta prochaine lecture grâce à une sélection personnalisée "
    "par intelligence artificielle."
)

st.caption(
    f"{len(catalogue)} livres disponibles dans le catalogue."
)

# ------------------------------------------------------------
# Suggestions d'accueil
# ------------------------------------------------------------

st.header("✨ Quelques idées pour commencer")

if "home_suggestions" not in st.session_state:
    st.session_state.home_suggestions = None

col_home1, col_home2 = st.columns([1, 4])

with col_home1:
    if st.button("🎲 Nouvelles idées", use_container_width=True):
        st.session_state.home_suggestions = None

if st.session_state.home_suggestions is None:
    with st.spinner("Recherche de deux idées de lecture..."):
        try:
            suggestions, home_usage = generate_home_suggestions(catalogue)
            st.session_state.home_suggestions = suggestions
            st.session_state.home_usage = home_usage
        except Exception as exc:
            st.warning(
                "Les suggestions d'accueil n'ont pas pu être générées. "
                "Tu peux tout de même demander une recommandation ci-dessous."
            )
            st.caption(f"Détail technique : {exc}")

if st.session_state.home_suggestions:
    home_cols = st.columns(2)

    for i, book in enumerate(st.session_state.home_suggestions):
        with home_cols[i]:
            st.markdown(f"### 📖 {book['titre']}")
            st.write(f"**Auteur :** {book['auteur']}")
            st.write(book["raison"])

st.divider()

# ------------------------------------------------------------
# Formulaire de recommandations
# ------------------------------------------------------------

st.header("🔎 Trouve un livre qui te correspond")

with st.form("recommendation_form"):
    genres = st.multiselect(
        "Quels genres aimerais-tu lire ?",
        [
            "Thriller",
            "Policier",
            "Suspense psychologique",
            "Roman contemporain",
            "Littérature générale",
            "Romance",
            "Science-fiction",
            "Fantasy",
            "Fantastique",
            "Historique",
            "Biographie",
            "Développement personnel",
            "Aventure",
            "Humour",
            "Feel-good",
        ],
    )

    mood = st.selectbox(
        "Quelle ambiance recherches-tu ?",
        [
            "Peu importe",
            "Captivante et pleine de suspense",
            "Sombre et inquiétante",
            "Émouvante",
            "Légère et divertissante",
            "Réconfortante",
            "Intellectuelle et profonde",
            "Dépaysante",
            "Romantique",
            "Surprenante avec des rebondissements",
        ],
    )

    length = st.selectbox(
        "Quelle longueur préfères-tu ?",
        [
            "Peu importe",
            "Court",
            "Moyen",
            "Long",
        ],
    )

    audience = st.selectbox(
        "Pour qui cherches-tu un livre ?",
        [
            "Pour moi",
            "Pour un adulte",
            "Pour un adolescent",
            "Pour un enfant",
            "Pour un lecteur de la famille",
        ],
    )

    exclusions = st.text_input(
        "Y a-t-il des genres, thèmes ou livres à éviter ?",
        placeholder="Ex. : pas de romance, pas de violence graphique...",
    )

    extra_details = st.text_area(
        "D'autres envies ou précisions ?",
        placeholder=(
            "Ex. : j'aime les intrigues avec un twist final, "
            "les personnages complexes..."
        ),
        height=100,
    )

    submitted = st.form_submit_button(
        "✨ Trouver mes recommandations",
        use_container_width=True,
    )

# ------------------------------------------------------------
# Résultats
# ------------------------------------------------------------

if submitted:
    with st.spinner(
        "L'IA analyse le catalogue complet et prépare ta sélection..."
    ):
        try:
            recommendations, external, usage, raw_text = recommend_books(
                catalogue,
                genres,
                mood,
                length,
                audience,
                exclusions,
                extra_details,
            )

            st.session_state.last_recommendations = recommendations
            st.session_state.last_external = external
            st.session_state.last_usage = usage
            st.session_state.last_raw = raw_text

        except Exception as exc:
            st.error(
                "Impossible d'obtenir les recommandations auprès d'OpenRouter."
            )
            st.code(str(exc))
            st.info(
                "Vérifie ta clé API, le nom du modèle, "
                "les quotas gratuits et la connexion."
            )

if "last_recommendations" in st.session_state:
    st.divider()
    st.header("📚 Voici ta sélection")

    recommendations = st.session_state.last_recommendations
    external = st.session_state.last_external

    if not recommendations:
        st.warning(
            "Aucune recommandation du catalogue n'a pu être validée. "
            "Essaie à nouveau avec des préférences plus simples."
        )

    for i, book in enumerate(recommendations, start=1):
        st.subheader(f"{i}. {book['titre']}")
        st.write(f"**Auteur :** {book['auteur']}")
        st.write(book["raison"] or "Une suggestion choisie pour tes préférences.")
        st.divider()

    if external:
        st.subheader("🌟 Une découverte en dehors de ton catalogue")
        st.markdown(f"### {external['titre']}")
        st.write(f"**Auteur :** {external['auteur']}")
        st.write(external["raison"])
    else:
        st.warning(
            "L'IA n'a pas fourni de recommandation extérieure valide "
            "dans cette réponse. Aucune deuxième requête automatique "
            "n'a été lancée."
        )

    usage = st.session_state.get("last_usage")
    if usage and usage.get("total_tokens") is not None:
        with st.expander("Détails techniques de la requête"):
            st.write(f"Modèle : `{MODEL}`")
            st.write(f"Tokens d'entrée : {usage.get('prompt_tokens')}")
            st.write(f"Tokens de sortie : {usage.get('completion_tokens')}")
            st.write(f"Total de tokens : {usage.get('total_tokens')}")

            st.caption(
                "Le total dépend notamment de la taille du catalogue "
                "et de la réponse générée."
            )
