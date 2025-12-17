import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal, Optional
import json
import os

# Load environment variables
load_dotenv()

# File path for saving question
QUESTION_FILE = "saved_question.json"

# Page config
st.set_page_config(
    page_title="Python Practice App",
    page_icon="📘",
    layout="centered"
)

# Helper functions
def save_question_to_file(question_data):
    with open(QUESTION_FILE, "w") as f:
        json.dump(question_data, f)

def load_question_from_file():
    if os.path.exists(QUESTION_FILE):
        with open(QUESTION_FILE, "r") as f:
            return json.load(f)
    return None

# Initialize session state
if 'generated_question' not in st.session_state:
    st.session_state.generated_question = None
if 'question_result' not in st.session_state:
    st.session_state.question_result = None


class pytem(BaseModel):
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Mention the difficulty level")
    Question: str = Field(description="Form the question")
    Snippet: Optional[str] = Field(description= "write short description of the python module with code example")
    Input: Optional[str] = Field(description="Mention the input variables")
    Output: Optional[str] = Field(description="Mention the output")

# Load saved question if available
saved_question = load_question_from_file()
if saved_question:
    st.session_state.generated_question = saved_question.get("Question")
    st.session_state.question_result = pytem(**saved_question)

# Sidebar
with st.sidebar:
    st.header("🛠️ Settings")
    st.write("Choose difficulty and subject to generate a Python practice question.")
    st.markdown("---")
    st.write("Made with ❤️ using LangChain + Gemini")

    if st.button("🔄 Reset Question"):
        if os.path.exists(QUESTION_FILE):
            os.remove(QUESTION_FILE)
        st.session_state.generated_question = None
        st.session_state.question_result = None
        st.success("Question reset successfully.")

# Title
st.markdown("<h1 style='text-align: center;'>🐍 Python Practice App</h1>", unsafe_allow_html=True)
st.markdown("### 🚀 Generate Python questions based on your selected topic and difficulty.")

# Select inputs
difficulty_level = st.selectbox("🎯 Select Difficulty Level", ['Easy', 'Medium', 'Hard'])


# ✅ Topic → Subtopic structure

topics = {
    "Core Python": [
        "String Manipulation", "List", "Dictionary", "Tuple", "Set", "List Comprehension",
        "Lambda Function", "Function", "Class", "Loop", "Exception Handling", "File I/O",
        "Regular Expressions", "Generators & Iterators", "Decorators", "Context Managers"
    ],
    "OOP": [
        "Encapsulation & Properties",
        "Inheritance (Single, Multiple)",
        "Polymorphism & Method Overriding",
        "Abstraction (ABC, Interfaces)",
        "Composition vs Aggregation",
        "Dataclasses & Immutability",
        "Mixins & Multiple Inheritance",
        "Dependency Injection & Inversion of Control",
        "Design Patterns (Factory, Strategy, Adapter)",
        "Solid Principles in Python"
    ],
    "Data Analysis": [
        "Numpy", "Pandas DataFrame", "Pandas Series", "Data Cleaning Techniques",
        "Feature Engineering", "Data Normalization & Scaling", "Exploratory Data Analysis (EDA)"
    ],
    "Visualization": [
        "Matplotlib", "Seaborn", "Plotly", "Altair"
    ],
    "Machine Learning": [
        "Machine Learning Basics", "Scikit-learn Pipelines", "Model Evaluation Metrics", "Hyperparameter Tuning"
    ],
    "Advanced Python": [
        "Multithreading & Multiprocessing", "Async Programming", "Virtual Environments & Package Management"
    ],
    "Data Handling & Integration": [
        "SQL Basics", "APIs & JSON Parsing", "Data Serialization"
    ],
    "Testing & Deployment": [
        "Unit Testing", "Logging", "Command-Line Arguments", "Environment Variables & Config Management"
    ],
    "Web & Automation": [
        "Web Scraping"
    ],
    "Databricks": [
        "Pyspark"
    ]
}


topic = st.selectbox("📂 Select Topic", list(topics.keys()))
subtopic = st.selectbox("📚 Select Subtopic", topics[topic])


snippet_required = st.selectbox("🎯 Whether you need snippet", ['Needed', 'Not needed'],index=1)

# Model setup
api_key = os.getenv("ANTHROPIC_API_KEY")


model = ChatAnthropic(model="claude-3-haiku-20240307", api_key=os.getenv("ANTHROPIC_API_KEY"), temperature=0.7)

# Pydantic schema for question generation
class pytem(BaseModel):
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Mention the difficulty level")
    Question: str = Field(description="Form the question")
    Snippet: Optional[str] = Field(description= "write short description of the python module with code example")
    Input: Optional[str] = Field(description="Mention the input variables")
    Output: Optional[str] = Field(description="Mention the output")

parser_py = PydanticOutputParser(pydantic_object=pytem)

# Prompt for question generation

if snippet_required == 'Needed':
    prompt_text = """
    Generate a Python practice question and a short tutorial snippet for the topic "{subject}" with difficulty level "{difficulty_level}".
    Return ONLY valid JSON in the following format:
    {format_instruction}
    """
else:
    prompt_text = """
    Generate ONLY a Python practice question for the topic "{subject}" with difficulty level "{difficulty_level}".
    Do NOT include any snippet.
    Return ONLY valid JSON in the following format:
    {format_instruction}
    """

prompt1 = PromptTemplate(
    input_variables=['difficulty_level', 'subject'],
    template= prompt_text,
    partial_variables={'format_instruction': parser_py.get_format_instructions()}
)


# Chain setup for question generation
chain1 = prompt1 | model | parser_py

# Generate question
if st.button('🧪 Generate Question'):
    result = chain1.invoke({'subject': subtopic, 'difficulty_level': difficulty_level})
    st.session_state.generated_question = result.Question
    st.session_state.question_result = result
    save_question_to_file(result.dict())  # Save to file

# Display question if available
if st.session_state.question_result:
    result = st.session_state.question_result
    col1, col2 = st.columns(2)

    with st.sidebar:
        st.markdown("Snippet")
        st.info(result.Snippet)

    with col1:
        st.markdown("### ❓ Question")
        st.info(result.Question)

        if result.Input:
            st.markdown("### 📥 Input")
            st.info(result.Input)

    with col2:
        st.markdown("### 📤 Expected Output")
        if result.Output:
            st.info(result.Output)

        st.markdown("### 🧠 Difficulty Level")
        st.success(result.difficulty)

# Answer input
st.markdown("---")
st.markdown("### ✍️ Submit Your Answer")
answer = st.text_area("Write your Python code here", height=200)

# Pydantic schema for answer checking
class pytem2(BaseModel):
    Verdict: Literal["Correct", "Wrong"] = Field(description="Mention the correctness of the code")
    Efficiency: int = Field(description="Give an efficiency score out of 100 measured against the most efficient code")
    Code: Optional[str] = Field(description="write the correct code for the question with highest efficiency")
    Why : Optional[str] = Field(description="Describe why the verdict is wrong in very SHORT and CRISP way, and only provide this if the verdict is wrong")

parser_py2 = PydanticOutputParser(pydantic_object=pytem2)

# Prompt for answer checking
prompt2 = PromptTemplate(
    input_variables=['question', 'answer'],
    template='Check the answer "{answer}" for the question "{question}" and give a structured judgement.\n{format_instruction}',
    partial_variables={'format_instruction': parser_py2.get_format_instructions()}
)

# Chain setup for answer checking
chain2 = prompt2 | model | parser_py2

# Check answer
if st.button('✅ Check Answer'):
    if st.session_state.generated_question:
        result2 = chain2.invoke({
            'answer': answer,
            'question': st.session_state.generated_question
        })

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🧾 Verdict")
            if result2.Verdict == 'Correct':
                st.markdown(f"<span style='color: green; font-weight: bold;'>{result2.Verdict}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: red; font-weight: bold;'>{result2.Verdict}</span>", unsafe_allow_html=True)

        with col2:
            st.markdown("### 📊 Efficiency Score")
            st.metric(label="Efficiency", value=f"{result2.Efficiency}%")

        if result2.Verdict == 'Correct' and result2.Code:
            st.markdown("### 💡 Suggested Better Code")
            st.code(result2.Code, language='python')

        if result2.Verdict == 'Wrong' and result2.Why:
            st.markdown("### 💡 Why it's wrong?")
            st.info(result2.Why)

    else:
        st.warning("Please generate a question first before checking your answer.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "© 2025 Python Practice App | Built by Anindya Sarkar"
    "</div>",
    unsafe_allow_html=True
)