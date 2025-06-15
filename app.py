import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

# Charger les artefacts
@st.cache_resource
def load_artefacts():
    model = tf.keras.models.load_model('dnn_model.keras')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    class_names = np.load('class_names.npy', allow_pickle=True)
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, le, class_names, feature_names

try:
    model, scaler, le, class_names, feature_names = load_artefacts()
    st.session_state.artefacts_loaded = True
except Exception as e:
    st.error(f"Erreur de chargement: {str(e)}")
    st.session_state.artefacts_loaded = False

# Interface utilisateur
st.set_page_config(
    page_title="Classifieur DNN",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🤖 Classification d'Appareils Électroniques")
st.write("Basé sur votre modèle DNN de Kaggle")

if st.session_state.get('artefacts_loaded', False):
    # Section d'entrée
    with st.form("input_form"):
        st.header("Caractéristiques de l'appareil")
        inputs = {}
        
        # Créer les champs dynamiquement
        cols = st.columns(3)
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                inputs[feature] = st.number_input(feature, value=0.0)
        
        submitted = st.form_submit_button("Prédire la catégorie")

    # Prédiction
    if submitted:
        try:
            # Créer le DataFrame
            input_df = pd.DataFrame([inputs])
            input_df = input_df[feature_names]
            
            # Prétraitement
            input_scaled = scaler.transform(input_df)
            
            # Prédiction
            prediction = model.predict(input_scaled)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            # Résultats
            st.success(f"**Catégorie prédite:** {predicted_class}")
            st.info(f"**Confiance:** {confidence:.2f}%")
            
            # Détails des probabilités
            st.subheader("Probabilités par catégorie")
            prob_df = pd.DataFrame({
                'Catégorie': class_names,
                'Probabilité': prediction[0]
            }).sort_values('Probabilité', ascending=False)
            
            # Affichage
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(prob_df.style.format({'Probabilité': '{:.2%}'}), height=300)
            with col2:
                st.bar_chart(prob_df.set_index('Catégorie'))
                
        except Exception as e:
            st.error(f"Erreur de prédiction: {str(e)}")
else:
    st.warning("Les artefacts n'ont pas pu être chargés. Vérifiez les fichiers.")

# Footer
st.divider()
st.caption("Projet Kaggle par [Votre Nom] | Déployé avec Streamlit")