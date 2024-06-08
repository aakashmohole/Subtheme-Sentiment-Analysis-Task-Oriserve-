
import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# Load the saved model
model = tf.keras.models.load_model('model.h5')



# Define a function to predict sentiment
def predict_sentiment(text):

    prediction = model.predict([text])[0][0]
    # Return sentiment prediction
    if prediction >= 0.5:
        return "Positive", prediction
    else:
        return "Negative", prediction

# Create a Gradio interface
iface = gr.Interface(fn=predict_sentiment, 
                     inputs=gr.Textbox(lines=2, placeholder='Comment to score'),  # Updated input component
                     outputs=gr.Textbox(),  # Specify output as a text box
                     title="Sentiment Analysis", 
                     description="Enter your review text and get sentiment prediction.")

# Launch the interface
iface.launch()