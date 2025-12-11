import streamlit as st
import numpy as np
from PIL import Image
import pickle

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Classifier")
st.write("This version uses only **Petal Length** and **Petal Width** for prediction.")

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model(filename="final_model_xgb.pkl"):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb')) # make sure the file is in repo

    return loaded_model  

model = load_model()

# ---------------------------------------------------------
# CLASS LABEL MAP
# ---------------------------------------------------------
species_map = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# ---------------------------------------------------------
# FLOWER IMAGES
# ---------------------------------------------------------
species_images = {
    "setosa": "images/setosa.jpg",
    "versicolor": "images/versicolor.jpg",
    "virginica": "images/virginica.jpg"
}

st.subheader("Iris Species Reference Images")
cols = st.columns(3)
labels = ["setosa", "versicolor", "virginica"]

for col, label in zip(cols, labels):
    col.image(
        Image.open(species_images[label]),
        caption=label.capitalize(),
        use_column_width=True
    )

st.markdown("---")

# ---------------------------------------------------------
# USER INPUT — 2 SLIDERS
# ---------------------------------------------------------
st.subheader("Enter Flower Measurements")

petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3, 0.1)
petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.3, 0.1)

# Only the 2 chosen features
features = np.array([[petal_length, petal_width]])

# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------

def load_pred_image(prediction):
    image_path = species_images[prediction]
    st.image(
        Image.open(image_path),
        caption=f"Predicted: {pred.capitalize()}",
        use_column_width=True
    )

# st.image(
#     Image.open(image_path),
#     caption=f"{pred.capitalize()}",
#     use_column_width=True
# )
if st.button("Predict Species"):
    pred_idx = model.predict(features)[0]     # integer
    pred = species_map[pred_idx]              # string
    
    prob = model.predict_proba(features)[0]

    st.markdown(f"### 🌿 Prediction: **{pred.capitalize()}**")

    st.write("Prediction Probabilities:")
    
    # Display as percentages in a formatted way
    prob_dict = {
        "setosa":     f"{prob[0]*100:.1f}%",
        "versicolor": f"{prob[1]*100:.1f}%",
        "virginica":  f"{prob[2]*100:.1f}%"
    }
    
    cols = st.columns(3)
    for col, (species, percentage) in zip(cols, prob_dict.items()):
        col.metric(label=species.capitalize(), value=percentage)

    load_pred_image(pred)
