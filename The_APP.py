import streamlit as st
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load inference results from the JSON file
with open("inference_results.json", "r") as f:
    inference_results = json.load(f)

# Aggregate all node inferences to create a graph-wide context
graph_context = " ".join([result["inference"] for result in inference_results])

# Load GPT-2 model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Streamlit App UI
st.title("Graph Knowledge Chat - Risk Management Insights")

# Input for User Question
user_question = st.text_input("Ask a question about risk management or the knowledge graph:", "What is risk management?")

# Button to Get Response
if st.button("Get Response"):
    # Truncate context if too large
    max_context_length = 600
    truncated_context = graph_context[:max_context_length]

    # Construct a prompt that uses the entire graph's context plus the user's question
    prompt = f"Context: {truncated_context}\nUser Question: {user_question}\nResponse:"

    try:
        # Encode the prompt and generate a response
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        max_new_tokens = 50  # Reduce to avoid memory issues
        output = gpt2_model.generate(inputs, max_new_tokens=max_new_tokens, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

        # Decode and display the GPT-2 response
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        st.subheader("Response")
        st.write(decoded_output)

    except RuntimeError as e:
        if "device-side assert triggered" in str(e):
            st.error("CUDA error detected. Try running on CPU or reduce input size.")
        else:
            st.error(f"An error occurred: {str(e)}")

st.write("Use the text input above to ask questions about risk management and gain insights from the knowledge graph.")
