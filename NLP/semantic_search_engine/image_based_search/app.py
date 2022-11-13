from html import escape
import re
import streamlit as st
import pandas as pd, numpy as np
import transformers
from transformers import CLIPProcessor, CLIPModel
from st_clickable_images import clickable_images

@st.cache(
    show_spinner=False,
    hash_funcs={
        CLIPModel: lambda _: None,
        CLIPProcessor: lambda _: None,
        dict: lambda _: None,
    },
)
def load():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    df = {0: pd.read_csv("data.csv"), 1: pd.read_csv("data2.csv")}
    embeddings = {0: np.load("embeddings.npy"), 1: np.load("embeddings2.npy")}
    for k in [0, 1]:
        embeddings[k] = embeddings[k] / np.linalg.norm(
            embeddings[k], axis=1, keepdims=True
        )
    return model, processor, df, embeddings


model, processor, df, embeddings = load()
source = {0: "\nSource: Unsplash", 1: "\nSource: The Movie Database (TMDB)"}


def compute_text_embeddings(list_of_strings):
    inputs = processor(text=list_of_strings, return_tensors="pt", padding=True)
    result = model.get_text_features(**inputs).detach().numpy()
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def image_search(query, corpus, n_results=24):
    positive_embeddings = None

    def concatenate_embeddings(e1, e2):
        if e1 is None:
            return e2
        else:
            return np.concatenate((e1, e2), axis=0)

    splitted_query = query.split("EXCLUDING ")
    dot_product = 0
    k = 0 if corpus == "Unsplash" else 1
    if len(splitted_query[0]) > 0:
        positive_queries = splitted_query[0].split(";")
        for positive_query in positive_queries:
            match = re.match(r"\[(Movies|Unsplash):(\d{1,5})\](.*)", positive_query)
            if match:
                corpus2, idx, remainder = match.groups()
                idx, remainder = int(idx), remainder.strip()
                k2 = 0 if corpus2 == "Unsplash" else 1
                positive_embeddings = concatenate_embeddings(
                    positive_embeddings, embeddings[k2][idx : idx + 1, :]
                )
                if len(remainder) > 0:
                    positive_embeddings = concatenate_embeddings(
                        positive_embeddings, compute_text_embeddings([remainder])
                    )
            else:
                positive_embeddings = concatenate_embeddings(
                    positive_embeddings, compute_text_embeddings([positive_query])
                )
        dot_product = embeddings[k] @ positive_embeddings.T
        dot_product = dot_product - np.median(dot_product, axis=0)
        dot_product = dot_product / np.max(dot_product, axis=0, keepdims=True)
        dot_product = np.min(dot_product, axis=1)

    if len(splitted_query) > 1:
        negative_queries = (" ".join(splitted_query[1:])).split(";")
        negative_embeddings = compute_text_embeddings(negative_queries)
        dot_product2 = embeddings[k] @ negative_embeddings.T
        dot_product2 = dot_product2 - np.median(dot_product2, axis=0)
        dot_product2 = dot_product2 / np.max(dot_product2, axis=0, keepdims=True)
        dot_product -= np.max(np.maximum(dot_product2, 0), axis=1)

    results = np.argsort(dot_product)[-1 : -n_results - 1 : -1]
    return [
        (
            df[k].iloc[i]["path"],
            df[k].iloc[i]["tooltip"] + source[k],
            i,
        )
        for i in results
    ]


description = """
# Semantic image search
**Enter your query and hit enter**
"""

howto = """
- Click image to find similar images
- Use "**;**" to combine multiple queries)
- Use "**EXCLUDING**", to exclude a query
"""


def main():
    st.markdown(
        """
              <style>
              .block-container{
                max-width: 1200px;
              }
              div.row-widget.stRadio > div{
                flex-direction:row;
                display: flex;
                justify-content: center;
              }
              div.row-widget.stRadio > div > label{
                margin-left: 5px;
                margin-right: 5px;
              }
              section.main>div:first-child {
                padding-top: 0px;
              }
              section:not(.main)>div:first-child {
                padding-top: 30px;
              }
              div.reportview-container > section:first-child{
                max-width: 320px;
              }
              #MainMenu {
                visibility: hidden;
              }
              footer {
                visibility: hidden;
              }
              </style>""",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(description)
    with st.sidebar.expander("Advanced use"):
        st.markdown(howto)


    st.sidebar.markdown(f"Unsplash has categories that match: backgrounds, photos, nature, iphone, etc")
    st.sidebar.markdown(f"Unsplash images contain animals, apps, events, feelings, food, travel, nature, people, religion, sports, things, stock")
    st.sidebar.markdown(f"Unsplash things include flag, tree, clock, money, tattoo, arrow, book, car, fireworks, ghost, health, kiss, dance, balloon, crown, eye, house, music, airplane, lighthouse, typewriter, toys")
    st.sidebar.markdown(f"unsplash feelings include funny, heart, love, cool, congratulations, love, scary, cute, friendship, inspirational, hug, sad, cursed, beautiful, crazy, respect, transformation, peaceful, happy")
    st.sidebar.markdown(f"unsplash people contain baby, life, women, family, girls, pregnancy, society, old people, musician, attractive, bohemian")
    st.sidebar.markdown(f"imagenet queries include: photo of, photo of many, sculpture of, rendering of, graffiti of, tattoo of, embroidered, drawing of, plastic, black and white, painting, video game, doodle, origami, sketch, etc")
    
    
    _, c, _ = st.columns((1, 3, 1))
    if "query" in st.session_state:
        query = c.text_input("", value=st.session_state["query"])
    else:

        query = c.text_input("", value="lighthouse")
    corpus = st.radio("", ["Unsplash"])
    #corpus = st.radio("", ["Unsplash", "Movies"])
    if len(query) > 0:
        results = image_search(query, corpus)
        clicked = clickable_images(
            [result[0] for result in results],
            titles=[result[1] for result in results],
            div_style={
                "display": "flex",
                "justify-content": "center",
                "flex-wrap": "wrap",
            },
            img_style={"margin": "2px", "height": "200px"},
        )
        if clicked >= 0:
            change_query = False
            if "last_clicked" not in st.session_state:
                change_query = True
            else:
                if clicked != st.session_state["last_clicked"]:
                    change_query = True
            if change_query:
                st.session_state["query"] = f"[{corpus}:{results[clicked][2]}]"
                st.experimental_rerun()


if __name__ == "__main__":
    main()