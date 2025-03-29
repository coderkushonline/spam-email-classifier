import streamlit as st
import joblib

# Loading the trained model
model = joblib.load('./spam_classifier.pkl')
feature_extraction = joblib.load('./email_to_number_feature_extraction.pkl')


def get_prediction(email: str):
       # Convert the input text to feature vectors
    input_data_features = feature_extraction.transform([email])
    
    # Make prediction
    prediction = model.predict(input_data_features)
    
    # Map prediction to 'Ham' or 'Spam'
    num_to_category = lambda x: "Ham" if x == 0 else "Spam" if x == 1 else "Error"
    return num_to_category(prediction[0])

st.title("Spam Mail Classifier")
if 'email' not in st.session_state:
    st.session_state.email = ""
st.text_input("Enter an email to classify it: ", value=st.session_state.email)

if st.button("Classify"):
    st.write(f"Email Type: {get_prediction(st.session_state.email)}")


st.divider()
st.subheader("Additional Information")
st.write("This app uses a pre-trained ML model to make its predictions. The link to the dataset used to train the model is given below:")
st.write("[Dataset Link](https://drive.google.com/file/d/1uzbhec5TW_OjFr4UUZkoMm0rpyvYdhZw/view)")