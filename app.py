import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from skimage.feature import hog

# ----------- CONFIG ----------- #
DATA_DIR = r'C:\smart_sorting_project\Fruit And Vegetable Diseases Dataset'
IMG_SIZE = 64  # Reduced for speed
MODEL_FILE = 'fruit_classifier.joblib'
ENCODER_FILE = 'label_encoder.joblib'
CACHE_FEATURES_FILE = 'features.npz'

# ----------- FEATURE EXTRACTION ----------- #
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, feature_vector=True)
    return features

# ----------- LOAD DATA OR CACHE ----------- #
def load_or_cache_features(max_images_per_class=20):
    if os.path.exists(CACHE_FEATURES_FILE):
        data = np.load(CACHE_FEATURES_FILE, allow_pickle=True)
        return data['X'], data['y']
    
    X, y = [], []

    for label in os.listdir(DATA_DIR):
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder): continue

        image_count = 0
        for img_file in os.listdir(folder):
            if image_count >= max_images_per_class:
                break
            img_path = os.path.join(folder, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                features = extract_features(img)
                X.append(features)
                y.append(label)
                image_count += 1
            except Exception as e:
                print(f"⚠️ Error processing {img_file}: {e}")
    
    np.savez_compressed(CACHE_FEATURES_FILE, X=X, y=y)
    return X, y

# ----------- TRAINING FUNCTION ----------- #
def train_model(max_images_per_class=20):
    st.info(f"Loading up to {max_images_per_class} images per class...")
    X, y = load_or_cache_features(max_images_per_class)

    if len(X) == 0:
        st.error("❌ No valid images found.")
        return "", None, None

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        labels=le.transform(le.classes_),
        target_names=le.classes_,
        zero_division=0
    )

    dump(model, MODEL_FILE)
    dump(le, ENCODER_FILE)

    return report, model, le

# ----------- PREPROCESS IMAGE ----------- #
def preprocess_image(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    features = extract_features(img_array)
    return features.reshape(1, -1)

# ----------- STREAMLIT UI ----------- #
st.set_page_config(page_title="Smart Sorting", layout="centered")
st.title("🍓 Smart Sorting - Fruit & Vegetable Classifier")
st.markdown("Upload a fruit/vegetable image to check if it's **Healthy** or **Rotten**.")

# Sidebar
st.sidebar.header("⚙️ Settings")
max_images = st.sidebar.slider("Images per class (training)", 5, 50, 20, step=5)
retrain = st.sidebar.checkbox("🔁 Force Retrain Model")

# Load or train
with st.spinner("Loading model..."):
    if retrain or not os.path.exists(MODEL_FILE) or not os.path.exists(ENCODER_FILE):
        report, model, le = train_model(max_images)
        if model:
            st.success("✅ Model trained.")
            st.text("Classification Report:\n" + report)
    else:
        model = load(MODEL_FILE)
        le = load(ENCODER_FILE)

# Upload & Predict
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    features = preprocess_image(img)
    prediction = model.predict(features)
    label = le.inverse_transform(prediction)[0]

    if '__' in label:
        fruit, condition = label.split('__')
        display_label = f"{fruit} — {condition}"
    else:
        display_label = label

    st.subheader(f"🔍 Prediction: **{display_label}**")
    if "Rotten" in label:
        st.warning("⚠️ This item appears to be **ROTTEN**.")
    else:
        st.success("✅ This item appears to be **HEALTHY**.")
