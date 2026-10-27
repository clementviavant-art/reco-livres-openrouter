
import os
import re
import random
import pandas as pd
import streamlit as st
from openai import OpenAI
from urllib.parse import quote_plus

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

MAIN_TOKENS = 2000
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
def babelio_search_url(title, author):
    """Ouvre Google sur une recherche du livre dans Babelio."""
    query = quote_plus(f'site:babelio.com/livres/ "{title}" "{author}"')
    return f"https://www.google.com/search?q={query}"
    
def normalize_text(text):
    """Normalise une chaîne pour comparer les titres."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def catalogue_for_prompt(df):
    """Prépare le catalogue compact transmis à l'IA."""

    lines = []

    for _, row in df.iterrows():
        lines.append(
            f"{int(row['ID'])}|{row['Auteur']}|{row['Titre du livre']}"
        )

    return "\n".join(lines)


def call_ai(system_prompt, user_prompt, max_tokens):
    """Appel unique à OpenRouter."""

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
                "effort": "none",
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


# ============================================================
# PARSING DES RECOMMANDATIONS
# ============================================================

def parse_recommendations(raw_text, df):
    """
    Accepte les formats :
    ID|identifiant|raison
    identifiant|auteur|raison
    EXT|titre|auteur|raison
    """

    by_id = {
        str(int(row["ID"])): row
        for _, row in df.iterrows()
    }

    catalogue_titles = {
        normalize_text(title)
        for title in df["Titre du livre"].tolist()
    }

    recommendations = []
    external = None
    seen_ids = set()

    for original_line in raw_text.splitlines():
        line = original_line.strip()

        if not line:
            continue

        line = re.sub(r"^[-*•\s]+", "", line)
        line = line.replace("`", "")

        parts = [part.strip() for part in line.split("|")]

        # Recommandation extérieure au catalogue
        if len(parts) >= 4 and parts[0].upper() == "EXT":
            title = parts[1]
            author = parts[2]
            reason = "|".join(parts[3:]).strip()

            normalized_title = normalize_text(title)

            if (
                title
                and author
                and normalized_title not in catalogue_titles
                and normalized_title not in {
                    normalize_text(item["titre"])
                    for item in recommendations
                }
            ):
                external = {
                    "titre": title,
                    "auteur": author,
                    "raison": reason,
                    "externe": True,
                }

            continue

        # Formats possibles pour le catalogue
        if len(parts) < 3:
            continue

        if parts[0].upper() == "ID":
            book_id = parts[1]
            reason = "|".join(parts[2:]).strip()

        elif parts[0].isdigit():
            book_id = parts[0]
            reason = "|".join(parts[2:]).strip()

        else:
            continue

        # Validation stricte de l'identifiant
        if book_id not in by_id or book_id in seen_ids:
            continue

        row = by_id[book_id]
        seen_ids.add(book_id)

        recommendations.append({
            "ID": int(book_id),
            "titre": row["Titre du livre"],
            "auteur": row["Auteur"],
            "raison": reason,
            "externe": False,
        })

    return recommendations[:3], external


# ============================================================
# SUGGESTIONS IA DE L'ACCUEIL
# ============================================================

def generate_home_suggestions(df):
    """Génère deux idées à partir d'un échantillon du catalogue."""

    sample_size = min(HOME_SAMPLE_SIZE, len(df))
    sample = df.sample(n=sample_size)

    lines = [
        f"{int(row['ID'])}|{row['Auteur']}|{row['Titre du livre']}"
        for _, row in sample.iterrows()
    ]

    system_prompt = """
Tu es un conseiller littéraire chaleureux et pertinent.

Choisis deux livres différents dans la liste fournie.
Utilise exclusivement les identifiants présents dans cette liste.

Réponds uniquement avec deux lignes, sans introduction ni conclusion.
Format obligatoire :
ID|identifiant|raison courte

La raison doit donner envie de découvrir le livre.
Chaque raison doit être courte.
N'ajoute aucun autre texte.
"""

    user_prompt = (
        "Voici les livres disponibles :\n"
        + "\n".join(lines)
        + "\n\nChoisis deux livres intéressants et variés."
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
            "Tu peux réessayer."
        )

    return selected[:2], usage


# ============================================================
# RECOMMANDATIONS PERSONNALISÉES
# ============================================================

def recommend_books(df, envie):
    """
    Envoie le catalogue complet et les envies exprimées en texte libre.
    Une seule requête sert à produire les recommandations principales.
    """

    full_catalogue = catalogue_for_prompt(df)

    system_prompt = """
Tu es un conseiller littéraire qui recommande des livres de manière personnalisée.

Tu dois proposer :
- 3 livres présents dans le catalogue fourni ;
- 1 livre supplémentaire qui ne figure pas dans le catalogue.

Pour chaque livre, rédige une présentation détaillée de 3 à 5 phrases.
Chaque présentation doit :
- expliquer précisément pourquoi ce livre correspond aux envies du lecteur ;
- présenter l'intrigue ou le sujet sans révéler les éléments importants de l'histoire ;
- décrire l'ambiance, le style ou le rythme de lecture ;
- expliquer ce qui pourrait particulièrement plaire au lecteur.

Évite les descriptions génériques. Chaque recommandation doit être personnalisée et apporter de vraies informations utiles pour choisir sa prochaine lecture.

Pour le livre supplémentaire, indique également son titre exact et son auteur.

Respecte impérativement le format suivant, avec une seule recommandation par ligne :

ID|identifiant|présentation détaillée
ID|identifiant|présentation détaillée
ID|identifiant|présentation détaillée
EXT|titre du livre|auteur|présentation détaillée

Règles importantes :
- Pour les 3 livres du catalogue, utilise uniquement les identifiants fournis.
- N'invente aucun identifiant.
- Le livre supplémentaire doit être différent des 3 livres du catalogue.
- N'ajoute aucun texte avant ou après les 4 lignes.
- N'utilise jamais le caractère | dans les présentations.
- Ne révèle pas la fin des livres ni les principaux retournements de situation.
"""

    user_prompt = f"""
Voici ce que le lecteur aimerait lire :

{envie}

CATALOGUE COMPLET
Format : ID|Auteur|Titre du livre

{full_catalogue}

Sélectionne les trois livres du catalogue les plus adaptés à cette demande,
puis propose un quatrième livre extérieur au catalogue.
"""

    raw_text, usage = call_ai(
        system_prompt,
        user_prompt,
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
    "Décris simplement ce que tu aimerais lire. "
    "L'intelligence artificielle cherchera les livres qui te correspondent."
)

st.caption(
    f"{len(catalogue)} livres disponibles dans le catalogue."
)


# ============================================================
# SUGGESTIONS D'ACCUEIL
# ============================================================

st.header("✨ Quelques idées pour commencer")

if "home_suggestions" not in st.session_state:
    st.session_state.home_suggestions = None

col_home1, col_home2 = st.columns([1, 4])

with col_home1:
    if st.button("🎲 Nouvelles idées", use_container_width=True):
        st.session_state.home_suggestions = None
        st.session_state.pop("home_error", None)

if st.session_state.home_suggestions is None:
    with st.spinner("Recherche de deux idées de lecture..."):
        try:
            suggestions, home_usage = generate_home_suggestions(catalogue)

            st.session_state.home_suggestions = suggestions
            st.session_state.home_usage = home_usage
            st.session_state.pop("home_error", None)

        except Exception as exc:
            st.session_state.home_error = str(exc)

if st.session_state.home_suggestions:
    home_cols = st.columns(2)

    for i, book in enumerate(st.session_state.home_suggestions):
        with home_cols[i]:
            st.markdown(f"### 📖 {book['titre']}")
            st.write(f"**Auteur :** {book['auteur']}")
            st.write(book["raison"])

elif st.session_state.get("home_error"):
    st.caption(
        "Les suggestions d'accueil ne sont pas disponibles pour le moment. "
        "Tu peux tout de même demander une recommandation ci-dessous."
    )

st.divider()


# ============================================================
# DEMANDE EN TEXTE LIBRE
# ============================================================

st.header("🔎 Quel livre te ferait envie ?")

with st.form("recommendation_form"):
    envie = st.text_area(
        "Décris tes envies de lecture",
        placeholder=(
            "Ex. : Je voudrais un thriller captivant avec des secrets "
            "de famille et une fin surprenante, pas trop long.\n\n"
            "Ou : J'ai adoré Harry Potter et je cherche un autre univers "
            "dans lequel me plonger.\n\n"
            "Ou encore : Je veux un roman émouvant, mais pas trop triste."
        ),
        height=150,
        label_visibility="collapsed",
    )

    submitted = st.form_submit_button(
        "✨ Trouver mes livres",
        use_container_width=True,
        type="primary",
    )


# ============================================================
# GÉNÉRATION DES RÉSULTATS
# ============================================================

if submitted:
    if not envie.strip():
        st.warning(
            "Écris quelques mots sur le livre que tu aimerais lire."
        )

    else:
        with st.spinner(
            "Je cherche les livres qui correspondent à tes envies..."
        ):
            try:
                recommendations, external, usage, raw_text = recommend_books(
                    catalogue,
                    envie.strip(),
                )

                st.session_state.last_recommendations = recommendations
                st.session_state.last_external = external
                st.session_state.last_usage = usage
                st.session_state.last_raw = raw_text
                st.session_state.last_envie = envie.strip()

            except Exception as exc:
                st.error(
                    "Impossible d'obtenir les recommandations auprès d'OpenRouter."
                )
                st.code(str(exc))
                st.info(
                    "Vérifie ta clé API, le nom du modèle, "
                    "les quotas gratuits et la connexion."
                )


# ============================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================

if "last_recommendations" in st.session_state:
    st.divider()
    st.header("📚 Voici ta sélection")

    recommendations = st.session_state.last_recommendations
    external = st.session_state.last_external

    if not recommendations:
        st.warning(
            "Aucune recommandation du catalogue n'a pu être validée. "
            "Essaie de reformuler ta demande."
        )

        with st.expander("Voir la réponse reçue"):
            st.text(st.session_state.get("last_raw", ""))

    for i, book in enumerate(recommendations, start=1):
        st.subheader(f"{i}. {book['titre']}")
        st.write(f"**Auteur :** {book['auteur']}")
        st.link_button(
            "📖 Voir sur Babelio",
            babelio_search_url(book["titre"], book["auteur"]),
        )
        st.write(
            book["raison"]
            or "Une suggestion choisie pour tes envies de lecture."
        )
        st.divider()

    if external:
        st.subheader("🌟 Une découverte en dehors de ton catalogue")
        st.markdown(f"### {external['titre']}")
        st.write(f"**Auteur :** {external['auteur']}")
        st.link_button(
            "📖 Voir sur Babelio",
            babelio_search_url(external["titre"], external["auteur"]),
        )
        st.write(external["raison"])

    else:
        st.warning(
            "La recommandation extérieure au catalogue n'a pas pu être "
            "validée dans cette réponse."
        )

        with st.expander("Voir la réponse brute de l'IA"):
            st.text(st.session_state.get("last_raw", ""))

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
