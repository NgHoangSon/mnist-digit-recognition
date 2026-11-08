import streamlit as st
from src.model import load_model
from src.preprocessing import preprocess_image
from src.inference import predict
# python -m pip install -r requirements.txt
st.set_page_config(
    page_title="MNIST Digit Regconition",
    page_icon="🔢",
    layout="centered"
)

st.title("MNIST Digit Regconition")

def get_model():
    return load_model()

model=get_model()


uploaded_file=st.file_uploader(
    "Choose an image",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:
    from PIL import Image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, 
             caption="Uploaded image", 
             width=250)

    try:
        x=preprocess_image(image)
        st.write("Input shape:", x.shape)
        st.image(
                x[0,0].numpy(),
                caption="Preprocessed image",
                clamp=True
            )
        pred,confidence,prob=predict(model,x)
        st.subheader("Prediction Result")
        col1,col2=st.columns(2)
        col1.metric("Prediction Digit", pred)
        col2.metric("Confidence",f"{confidence:.2%}")

    
        # Hiển thị xác suất của 10 chữ số
        st.subheader("Probability Distribution")

        probability_data = {
            "Digit": list(range(10)),
            "Probability": prob[0].tolist()
        }

        st.bar_chart(
            probability_data,
            x="Digit",
            y="Probability"
        )
    except ValueError as e:
        st.error(str(e))
