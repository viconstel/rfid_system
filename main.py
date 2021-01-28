import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics import pairwise_distances
from PIL import Image
from imageio import imread


VECTORIZER_PATH = './title_vectorizer.pkl'
DATAFRAME_PATH = './data_with_big_url.pkl'
WRAPPER = '<span style="font-size:1.5em">{0}</span>'
BREAK = '<br>'
ROWS, COLS = 4, 5
IMAGE_SIZE = 256, 256


class Model:
    def __init__(self):
        self.data = self.load_data()
        self.vectorizer = self.load_vectorizer()

    @staticmethod
    def load_data():
        return pd.read_pickle(DATAFRAME_PATH)

    @staticmethod
    def load_vectorizer():
        return joblib.load(VECTORIZER_PATH)

    def predict(self, item_id, num_results=5):
        pairwise_dist = pairwise_distances(self.vectorizer,
                                           self.vectorizer[item_id])

        # np.argsort will return indices of the smallest distances
        indices = np.argsort(pairwise_dist.flatten())[1:1 + num_results]
        # dists will store the smallest distances
        dists = np.sort(pairwise_dist.flatten())[1:1 + num_results]

        return indices


@st.cache
def get_image(url):
    im = Image.fromarray(imread(url))
    im = im.resize(IMAGE_SIZE, Image.ANTIALIAS)
    return im


def local_css(file_name: str) -> None:
    """Setup CSS style."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def main():
    model = Model()
    st.set_page_config(layout='wide')
    local_css('./style.css')
    categories = ['Shirt', 'Accessory', 'Sporting Goods', 'Dress', 'Blazer',
                  'Sweater', 'Outerwear', 'Outdoor Recreation Product',
                  'Skirt', 'Sleepwear']
    radio = st.sidebar.radio('Select a section:', ['Home'] + categories)

    if radio == 'Home':
        st.markdown("# 👗 Clothing recommendation system" + BREAK,
                    unsafe_allow_html=True)
        st.markdown(WRAPPER.format("This is a demo service of a content-based "
                                   "recommendation system for clothing "
                                   "recommendations. The model is based on a "
                                   "Amazon Women Apparel dataset. Select a "
                                   "section to get started!") + 2 * BREAK,
                    unsafe_allow_html=True)
    else:
        st.markdown(WRAPPER.format("⬇️ **Select a thing and scroll down the "
                                   "page to see recommendations**") + 2 * BREAK,
                    unsafe_allow_html=True)
        cat = '_'.join(radio.split()).upper()
        items = model.data[model.data.product_type_name == cat][:ROWS * COLS]
        keys_dict = {}

        for i in range(ROWS):
            c1, c2, c3, c4, c5 = st.beta_columns(COLS)

            with c1:
                item = items.iloc[i * COLS, :]
                url = item.big_url
                title = item.old_title
                brand, asin, color = item.brand, item.asin, item.color
                st.image(get_image(url), use_column_width=True)
                with st.beta_expander('See options', expanded=False):
                    st.write(f'**Title:** {title}')
                    st.write(f'**Brand:** {brand}')
                    st.write(f'**Type:** {radio}')
                    st.write(f'**Color:** {color}')
                    st.write(f'**ASIN:** {asin}')
                st.write(BREAK, unsafe_allow_html=True)
                button = st.button('Select!', key=asin)
                keys_dict[asin] = button
            with c2:
                item = items.iloc[i * COLS + 1, :]
                url = item.big_url
                title = item.old_title
                brand, asin, color = item.brand, item.asin, item.color
                st.image(get_image(url), use_column_width=True)
                with st.beta_expander('See options', expanded=False):
                    st.write(f'**Title:** {title}')
                    st.write(f'**Brand:** {brand}')
                    st.write(f'**Type:** {radio}')
                    st.write(f'**Color:** {color}')
                    st.write(f'**ASIN:** {asin}')
                st.write(BREAK, unsafe_allow_html=True)
                button = st.button('Select!', key=asin)
                keys_dict[asin] = button
            with c3:
                item = items.iloc[i * COLS + 2, :]
                url = item.big_url
                title = item.old_title
                brand, asin, color = item.brand, item.asin, item.color
                st.image(get_image(url), use_column_width=True)
                with st.beta_expander('See options', expanded=False):
                    st.write(f'**Title:** {title}')
                    st.write(f'**Brand:** {brand}')
                    st.write(f'**Type:** {radio}')
                    st.write(f'**Color:** {color}')
                    st.write(f'**ASIN:** {asin}')
                st.write(BREAK, unsafe_allow_html=True)
                button = st.button('Select!', key=asin)
                keys_dict[asin] = button
            with c4:
                item = items.iloc[i * COLS + 3, :]
                url = item.big_url
                title = item.old_title
                brand, asin, color = item.brand, item.asin, item.color
                st.image(get_image(url), use_column_width=True)
                with st.beta_expander('See options', expanded=False):
                    st.write(f'**Title:** {title}')
                    st.write(f'**Brand:** {brand}')
                    st.write(f'**Type:** {radio}')
                    st.write(f'**Color:** {color}')
                    st.write(f'**ASIN:** {asin}')
                st.write(BREAK, unsafe_allow_html=True)
                button = st.button('Select!', key=asin)
                keys_dict[asin] = button
            with c5:
                item = items.iloc[i * COLS + 4, :]
                url = item.big_url
                title = item.old_title
                brand, asin, color = item.brand, item.asin, item.color
                st.image(get_image(url), use_column_width=True)
                with st.beta_expander('See options', expanded=False):
                    st.write(f'**Title:** {title}')
                    st.write(f'**Brand:** {brand}')
                    st.write(f'**Type:** {radio}')
                    st.write(f'**Color:** {color}')
                    st.write(f'**ASIN:** {asin}')
                st.write(BREAK, unsafe_allow_html=True)
                button = st.button('Select!', key=asin)
                keys_dict[asin] = button
            st.write(2 * BREAK, unsafe_allow_html=True)

        for key in keys_dict.keys():
            if keys_dict[key]:
                ind = model.data[model.data.asin == key].index[0]
                indices = model.predict(ind)
                st.markdown('# Recommended goods:' + BREAK,
                            unsafe_allow_html=True)
                st.write(WRAPPER.format('Showing top-5 recommendations' + BREAK),
                         unsafe_allow_html=True)
                c1, c2, c3, c4, c5 = st.beta_columns(COLS)

                with c1:
                    item = model.data.iloc[indices[0]]
                    categ = item.product_type_name.split('_')
                    categ = ' '.join([i[0] + i[1:].lower() for i in categ])
                    url = item.big_url
                    title = item.old_title
                    brand, asin, color = item.brand, item.asin, item.color
                    st.image(get_image(url), use_column_width=True)
                    with st.beta_expander('See options', expanded=False):
                        st.write(f'**Title:** {title}')
                        st.write(f'**Brand:** {brand}')
                        st.write(f'**Type:** {categ}')
                        st.write(f'**Color:** {color}')
                        st.write(f'**ASIN:** {asin}')
                with c2:
                    item = model.data.iloc[indices[1]]
                    categ = item.product_type_name.split('_')
                    categ = ' '.join([i[0] + i[1:].lower() for i in categ])
                    url = item.big_url
                    title = item.old_title
                    brand, asin, color = item.brand, item.asin, item.color
                    st.image(get_image(url), use_column_width=True)
                    with st.beta_expander('See options', expanded=False):
                        st.write(f'**Title:** {title}')
                        st.write(f'**Brand:** {brand}')
                        st.write(f'**Type:** {categ}')
                        st.write(f'**Color:** {color}')
                        st.write(f'**ASIN:** {asin}')
                with c3:
                    item = model.data.iloc[indices[2]]
                    categ = item.product_type_name.split('_')
                    categ = ' '.join([i[0] + i[1:].lower() for i in categ])
                    url = item.big_url
                    title = item.old_title
                    brand, asin, color = item.brand, item.asin, item.color
                    st.image(get_image(url), use_column_width=True)
                    with st.beta_expander('See options', expanded=False):
                        st.write(f'**Title:** {title}')
                        st.write(f'**Brand:** {brand}')
                        st.write(f'**Type:** {categ}')
                        st.write(f'**Color:** {color}')
                        st.write(f'**ASIN:** {asin}')
                with c4:
                    item = model.data.iloc[indices[3]]
                    categ = item.product_type_name.split('_')
                    categ = ' '.join([i[0] + i[1:].lower() for i in categ])
                    url = item.big_url
                    title = item.old_title
                    brand, asin, color = item.brand, item.asin, item.color
                    st.image(get_image(url), use_column_width=True)
                    with st.beta_expander('See options', expanded=False):
                        st.write(f'**Title:** {title}')
                        st.write(f'**Brand:** {brand}')
                        st.write(f'**Type:** {categ}')
                        st.write(f'**Color:** {color}')
                        st.write(f'**ASIN:** {asin}')
                with c5:
                    item = model.data.iloc[indices[4]]
                    categ = item.product_type_name.split('_')
                    categ = ' '.join([i[0] + i[1:].lower() for i in categ])
                    url = item.big_url
                    title = item.old_title
                    brand, asin, color = item.brand, item.asin, item.color
                    st.image(get_image(url), use_column_width=True)
                    with st.beta_expander('See options', expanded=False):
                        st.write(f'**Title:** {title}')
                        st.write(f'**Brand:** {brand}')
                        st.write(f'**Type:** {categ}')
                        st.write(f'**Color:** {color}')
                        st.write(f'**ASIN:** {asin}')


if __name__ == '__main__':
    main()
