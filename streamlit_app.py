import streamlit as st
import google.generativeai as genai
import sympy
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("The Quantum Leap Calculator: Ever AI Edition")
st.write("Unleash the power of AI-driven calculations, visualizations, and mind-bending functions!")

# Model Initialization
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to execute and display SymPy calculations
def execute_sympy(expression):
    try:
        result = sympy.sympify(expression)
        st.write("Result:")
        st.latex(sympy.latex(result))
    except Exception as e:
        st.error(f"SymPy Error: {e}")

# Function to plot graphs
def plot_graph(expression, x_range=(-10, 10)):
    try:
        x = sympy.symbols('x')
        expr = sympy.sympify(expression)
        f = sympy.lambdify(x, expr, 'numpy')
        x_vals = np.linspace(x_range[0], x_range[1], 400)
        y_vals = f(x_vals)

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Graph of {expression}')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Plotting Error: {e}")

# Function to generate and display images based on text prompts
def generate_image(prompt):
    try:
        model_image = genai.GenerativeModel('gemini-pro-vision')
        response = model_image.generate_content(prompt)
        if response.parts:
            for part in response.parts:
                if part.inline_data and part.inline_data.mime_type == "image/png":
                    image_bytes = part.inline_data.data
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image, caption="Generated Image", use_column_width=True)
                else:
                    st.write("No image found in response.")
        else:
            st.write("No parts in response.")
    except Exception as e:
        st.error(f"Image Generation Error: {e}")

# Tabbed Interface for different features
tab1, tab2, tab3, tab4 = st.tabs(["AI Math", "AI Graphing", "AI Image Generation", "AI General"])

with tab1:
    st.header("AI Powered Mathematical Calculations")
    math_input = st.text_input("Enter a mathematical expression (e.g., sin(pi/4) + sqrt(16)):", "2 + 2")
    if st.button("Calculate"):
        execute_sympy(math_input)

with tab2:
    st.header("AI Graphing Calculator")
    graph_input = st.text_input("Enter a function to graph (e.g., x^2 + 2x + 1):", "sin(x)")
    x_min, x_max = st.slider("X-axis range", -20, 20, (-10, 10))
    if st.button("Plot Graph"):
        plot_graph(graph_input, (x_min, x_max))

with tab3:
    st.header("AI Image Generator")
    image_prompt = st.text_input("Enter a prompt for image generation:", "A futuristic cat riding a unicorn in space")
    if st.button("Generate Image"):
        generate_image(image_prompt)

with tab4:
    st.header("AI General Prompt")
    general_prompt = st.text_area("Enter your general AI prompt:", "Explain quantum entanglement in simple terms.")
    if st.button("Get AI Response"):
        try:
            response = model.generate_content(general_prompt)
            st.write("AI Response:")
            st.write(response.text)
        except Exception as e:
            st.error(f"AI Response Error: {e}")
