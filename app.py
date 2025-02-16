# -*- coding: utf-8 -*-

import os
import gradio as gr
import whisper
from gtts import gTTS
from groq import Groq
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

index_file_path="faiss_index.index"
embeddings_file_path="embeddings.npy"
# Load Whisper model for transcription
model = whisper.load_model("base")

# Set up Groq API client (make sure your API key is correct)
client = Groq(api_key="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")

# Load the dataset
df = pd.read_json("hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json", lines=True)


corpus = df['Context'].dropna().tolist()

# Initialize SentenceTransformer to generate embeddings
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to load or build the FAISS index
def load_or_build_index():
    if os.path.exists(index_file_path) and os.path.exists(embeddings_file_path):
        print("Loading existing index and embeddings...")
        index = faiss.read_index(index_file_path)
        embeddings = np.load(embeddings_file_path)
    else:
        print("Building new index...")
        embeddings = embedder.encode(corpus, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # FAISS index for L2 (Euclidean) distance
        index.add(embeddings)
        faiss.write_index(index, index_file_path)  # Save the index to disk
        np.save(embeddings_file_path, embeddings)  # Save embeddings to disk
    return index, embeddings

# Load or build the FAISS index
index, corpus_embeddings = load_or_build_index()

# Function to retrieve the most relevant context from the corpus using FAISS
def retrieve_relevant_context(user_input):
    user_input_embedding = embedder.encode([user_input])  # Convert the user's query into an embedding
    k = 1  # Retrieve the top 1 most relevant document
    D, I = index.search(user_input_embedding, k)  # Perform the search in the FAISS index
    return corpus[I[0][0]]  # Return the most relevant document

# Function to process the audio input, retrieve context, and generate a response
def chatbot(audio):
    # Transcribe the audio input using Whisper
    transcription = model.transcribe(audio)
    user_input = transcription["text"]

    # Retrieve the most relevant context from the dataset using the vector database (FAISS)
    relevant_context = retrieve_relevant_context(user_input)

    # Generate a response using the Groq API with Llama 8B, including relevant context
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": user_input},
            {"role": "system", "content": f"Context: {relevant_context}"}
        ],
        model="llama3-8b-8192"
    )
    
    # Extract the generated response text
    response_text = chat_completion.choices[0].message.content

    # Convert the response text to speech using gTTS
    tts = gTTS(text=response_text, lang='en')
    tts.save("response.mp3")

    return response_text, "response.mp3"

# Create a custom Gradio interface
def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            <h1 style="text-align: center; color: #4CAF50;">Chill Parents</h1>
            <h3 style="text-align: center;">Chatbot to help parents and other family members to reduce stress between them</h3>
            <p style="text-align: center;">Talk to the AI-powered chatbot and get responses in real-time. Start by recording your voice.</p>
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(type="filepath", label="Record Your Voice")
            with gr.Column(scale=2):
                chatbot_output_text = gr.Textbox(label="Chatbot Response")
                chatbot_output_audio = gr.Audio(label="Audio Response")

        submit_button = gr.Button("Submit")

        submit_button.click(chatbot, inputs=audio_input, outputs=[chatbot_output_text, chatbot_output_audio])

    return demo

# Launch the interface
if __name__ == "__main__":
    interface = build_interface()
    interface.launch()
