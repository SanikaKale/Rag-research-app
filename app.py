import streamlit as st
import uuid
import os
import ast
import re
import urllib.parse
from urllib.parse import quote
import pandas as pd
from transformers import pipeline


# --- Helper Functions for URI and MeSH Handling (from notebook) ---
def parse_mesh_terms(mesh_list):
    if pd.isna(mesh_list):
        return []
    return [term.strip() for term in mesh_list.strip("[]'").split(',')]

def convert_to_uri(term, base_namespace="http://example.org/mesh/"):
    if pd.isna(term):
        return None
    stripped_term = re.sub(r'^\W+|\W+$', '', term)
    formatted_term = re.sub(r'\W+', '_', stripped_term)
    formatted_term = re.sub(r'_+', '_', formatted_term)
    encoded_term = quote(formatted_term)
    term_with_underscores = f"_{encoded_term}_"
    uri = f"{base_namespace}{term_with_underscores}"
    return uri

def create_article_uri(title, base_namespace="http://example.org/article"):
    if pd.isna(title):
        return None
    sanitized_text = urllib.parse.quote(title.strip().replace(' ', '_').replace('"', '').replace('<', '').replace('>', '').replace("'", "_"))
    return f"{base_namespace}/{sanitized_text}"

# --- HuggingFace Summarizer (Free LLM) ---
@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = get_summarizer()

def generate_summary_with_hf(text, user_query=None):
    input_text = text if not user_query else f"{user_query}\n\n{text}"
    max_chunk = 1024
    if len(input_text) > max_chunk:
        input_text = input_text[:max_chunk]
    summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    return summary

def generate_multi_summary(articles, user_query=None):
    summaries = []
    for article_uri, data in articles:
        text = f"Title: {data['title']}\nAbstract: {data['abstract']}"
        if user_query:
            text = f"{user_query}\n\n{text}"
        # Truncate if needed
        text = text[:1024]
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(f"**{data['title']}**: {summary}")
    return "\n\n".join(summaries)


# --- Initialization ---
if "article_results" not in st.session_state:
    st.session_state.article_results = []
if "selected_terms" not in st.session_state:
    st.session_state.selected_terms = {}
if "expanded_terms" not in st.session_state:
    st.session_state.expanded_terms = {}
if "current_search_terms" not in st.session_state:
    st.session_state.current_search_terms = []
if "search_session_id" not in st.session_state:
    st.session_state.search_session_id = 0
if "node_registry" not in st.session_state:
    st.session_state.node_registry = {}
if "node_data" not in st.session_state:
    st.session_state.node_data = {}
if "node_widget_ids" not in st.session_state:
    st.session_state.node_widget_ids = {}
if "node_counter" not in st.session_state:
    st.session_state.node_counter = 0

st.title("RAG for Research")
st.subheader("Retrieval-Augmented Generation for Medical Journal Articles")

tabs = st.tabs(["1. Search Articles", "2. Refine Terms", "3. Filter-Summarize"])
tab_search, tab_refine, tab_filter = tabs

# --- TAB 1: Search Articles ---
with tab_search:
    st.header("Search Articles ")
    query_text = st.text_input("Enter your vector search term (e.g., Carcinoma):", key="vector_search")

    # Stub: Replace with your own article search logic or load from a CSV
    if st.button("Search Articles", key="search_articles_btn"):
        try:
            # Example: Load from a CSV (replace with your own logic)
            df = pd.read_csv("PubMed Multi Label Text Classification Dataset.csv")
            # Filter by query_text in title or abstract
            article_results = df[df['Title'].str.contains(query_text, case=False, na=False) | df['abstractText'].str.contains(query_text, case=False, na=False)].head(10)
            article_uris = [create_article_uri(row['Title']) for _, row in article_results.iterrows()]
            st.session_state.article_uris = article_uris
            st.session_state.article_results = [
                {
                    "Title": row["Title"],
                    "Abstract": (row["abstractText"][:100] + "...") if len(row["abstractText"]) > 100 else row["abstractText"],
                    "MeSH Terms": ", ".join(parse_mesh_terms(row["meshMajor"])),
                }
                for _, row in article_results.iterrows()
            ]
        except Exception as e:
            st.error(f"Error during article search: {e}")

    if st.session_state.article_results:
        st.write("**Search Results:**")
        st.table(st.session_state.article_results)
    else:
        st.write("No articles.")

def get_node_id(term, path):
    key = (term, tuple(path), st.session_state.search_session_id)
    if key not in st.session_state.node_registry:
        st.session_state.node_registry[key] = st.session_state.node_counter
        st.session_state.node_counter += 1
    return st.session_state.node_registry[key]

# --- Helper Function for Recursive Display ---
def display_term(term, path=None, visited=None, level=0):
    if path is None:
        path = []
    if visited is None:
        visited = set()

    node_id = get_node_id(term, path)

    if node_id in visited:
        indent = "&emsp;" * (level * 4)
        st.markdown(f"{indent}_Already displayed {term}, skipping._", unsafe_allow_html=True)
        return
    visited.add(node_id)

    indent = "&emsp;" * (level * 4)
    prefix = "" if level == 0 else "└─ "

    path_str = "_".join(path)
    term_key = f"cb_{node_id}"
    expand_button_key = f"expand_{node_id}"

    if term not in st.session_state.selected_terms:
        st.session_state.selected_terms[term] = False

    st.session_state.selected_terms[term] = st.checkbox(
        f"{indent}{prefix}{term}",
        value=st.session_state.selected_terms[term],
        key=term_key
    )

    if node_id not in st.session_state.node_data:
        st.session_state.node_data[node_id] = {
            "term": term,
            "alt_names": [],
            "narrower_concepts": {},
            "expanded": False
        }

    expanded = st.session_state.node_data[node_id]["expanded"]
    expand_label = "Collapse" if expanded else "Expand"

    if st.button(f"{indent}{prefix}{expand_label} {term}", key=expand_button_key):
        if expanded:
            st.session_state.node_data[node_id]["expanded"] = False
        else:
            # Stub: Replace with your own logic to fetch alt_names and narrower_concepts
            alt_names = []
            narrower_concepts = {}
            st.session_state.node_data[node_id]["alt_names"] = alt_names
            st.session_state.node_data[node_id]["narrower_concepts"] = narrower_concepts
            st.session_state.node_data[node_id]["expanded"] = True
            expanded = True

    if expanded:
        alt_names = st.session_state.node_data[node_id]["alt_names"]
        narrower_concepts = st.session_state.node_data[node_id]["narrower_concepts"]

        if alt_names:
            st.markdown(f"{indent}**Alternative Names:**", unsafe_allow_html=True)
            for alt_name in alt_names:
                alt_path = path + [term, "alt"]
                alt_path_str = "_".join(alt_path)
                child_id = get_node_id(alt_name, alt_path)
                alt_key = f"alt_{child_id}"
                if alt_name not in st.session_state.selected_terms:
                    st.session_state.selected_terms[alt_name] = False
                st.session_state.selected_terms[alt_name] = st.checkbox(
                    f"{indent}&emsp;&emsp;• {alt_name}",
                    value=st.session_state.selected_terms[alt_name],
                    key=alt_key
                )

        if narrower_concepts:
            st.markdown(f"{indent}**Narrower Concepts:**", unsafe_allow_html=True)
            for narrower, children in narrower_concepts.items():
                st.markdown(f"{indent}&emsp;• **{narrower}**", unsafe_allow_html=True)
                for child in children:
                    display_term(child, path=path+[term, narrower], visited=visited, level=level+1)

# --- TAB 2: Refine Terms ---
with tab_refine:
    st.header("Refine Terms for Filtering")
    mesh_query_text = st.text_input("Enter a term for refinement:", key="mesh_search_input")

    if st.button("Search Terms", key="search_mesh_terms_btn"):
        st.session_state.search_session_id += 1
        try:
            st.session_state.current_search_terms.clear()
            st.session_state.node_registry = {}
            st.session_state.node_data = {}
            st.session_state.node_counter = 0
            # Stub: Replace with your own logic to search MeSH terms
            # Example: Load from a CSV and filter
            df = pd.read_csv("PubMed Multi Label Text Classification Dataset.csv")
            mesh_terms = set()
            for mesh_list in df["meshMajor"]:
                mesh_terms.update(parse_mesh_terms(mesh_list))
            filtered_terms = [t for t in mesh_terms if mesh_query_text.lower() in t.lower()]
            st.session_state.current_search_terms = filtered_terms[:10]
            for term in st.session_state.current_search_terms:
                if term not in st.session_state.selected_terms:
                    st.session_state.selected_terms[term] = False
        except Exception as e:
            st.error(f"Error during term search: {e}")

    if st.session_state.current_search_terms:
        st.subheader("Current Search Results for MeSH Terms")
        st.write("Select terms and expand them to deeper on the concepts.")
        for term in st.session_state.current_search_terms:
            display_term(term, path=[term], visited=set(), level=0)
    else:
        st.write("No current search results.")

# SIDEBAR
with st.sidebar:
    st.header("Selected Terms")
    selected_display = [t for t, selected in st.session_state.selected_terms.items() if selected]
    if selected_display:
        st.write(", ".join(selected_display))
    else:
        st.write("No terms selected yet.")
    st.write("---")
    st.write("**Instructions:**")
    st.write("1. 'Search Articles' to find relevant articles.")
    st.write("2. 'Refine Terms' to find and select terms and expand them.")
    st.write("3. 'Filter & Summarize' to apply filters then get summaries.")

# --- TAB 3: Filter & Summarize ---
with tab_filter:
    st.header("Filter and Summarize Results")
    final_terms = [t for t, selected in st.session_state.selected_terms.items() if selected]
    LOCAL_FILE_PATH = "PubMedGraph.ttl"

    if final_terms:
        st.write("**Final Terms for Filtering:**")
        st.write(", ".join(final_terms))

        # Stub: Download RDF file if needed (skip if already present)
        if "rdf_file_downloaded" not in st.session_state:
            try:
                # If you have a download function, call it here. Otherwise, check file exists.
                if not os.path.exists(LOCAL_FILE_PATH):
                    st.error(f"RDF file {LOCAL_FILE_PATH} not found. Please generate it.")
                    st.stop()
                st.session_state.rdf_file_downloaded = True
            except Exception as e:
                st.error(f"Error downloading RDF file: {e}")

        if st.button("Filter Articles"):
            try:
                if "article_uris" in st.session_state and st.session_state.article_uris:
                    article_uris = st.session_state.article_uris
                    article_uris_string = ", ".join([f"<{str(uri)}>" for uri in article_uris])
                    SPARQL_QUERY = f"""
                    PREFIX schema: <http://schema.org/>
                    PREFIX ex: <http://example.org/>
                    SELECT ?article ?title ?abstract ?datePublished ?access ?meshTerm
                    WHERE {{
                      ?article a ex:Article ;
                               schema:name ?title ;
                               schema:description ?abstract ;
                               schema:datePublished ?datePublished ;
                               ex:access ?access ;
                               schema:about ?meshTerm .
                      ?meshTerm a ex:MeSHTerm .
                      FILTER (?article IN ({article_uris_string}))
                    }}
                    """
                    # Stub: Replace with your own RDF query logic
                    # For now, just load from CSV and filter
                    df = pd.read_csv("PubMed Multi Label Text Classification Dataset.csv")
                    filtered = df[df['Title'].apply(lambda t: create_article_uri(t) in article_uris)]
                    top_articles = []
                    for _, row in filtered.iterrows():
                        mesh_terms = parse_mesh_terms(row['meshMajor'])
                        if any(term in mesh_terms for term in final_terms):
                            top_articles.append((create_article_uri(row['Title']), {
                                'title': row['Title'],
                                'abstract': row['abstractText'],
                                'meshTerms': mesh_terms
                            }))
                    st.session_state.filtered_articles = top_articles
                    if top_articles:
                        def combine_abstracts(ranked_articles):
                            combined_text = " ".join(
                                [f"Title: {data['title']} Abstract: {data['abstract']}" for article_uri, data in ranked_articles]
                            )
                            return combined_text
                        st.session_state.combined_text = combine_abstracts(top_articles)
                    else:
                        st.write("No articles found for the selected terms.")
                else:
                    st.write("No articles selected from Tab 1.")
                    st.stop()
            except Exception as e:
                st.error(f"Error filtering articles: {e}")

        if "user_query" not in st.session_state:
            st.session_state.user_query = "Summarize the key information."

        st.session_state.user_query = st.text_area(
            "Enter your query for the LLM:",
            value=st.session_state.user_query,
            key="user_query_text_area",
            height=100,
        )

        if "filtered_articles" in st.session_state and st.session_state.filtered_articles:
            st.subheader("Original Articles")
            for article_uri, data in st.session_state.filtered_articles:
                st.write(f"**Title:** {data['title']}")
                st.write(f"**Abstract:** {data['abstract']}")
                st.write("**MeSH Terms:**")
                for mesh_term in data['meshTerms']:
                    st.write(f"- {mesh_term}")
                st.write("---")

        if st.button("Summarize with LLM"):
            try:
                if "filtered_articles" in st.session_state and st.session_state["filtered_articles"]:
                    user_query = st.session_state.user_query
                    with st.spinner("Generating summary..."):
                        summary = generate_multi_summary(st.session_state["filtered_articles"], user_query)
                    st.subheader("Summary")
                    st.write(summary)
                else:
                    st.error("No text available for summarization. Please filter articles first.")
            except Exception as e:
                st.error(f"Error summarizing articles: {e}")