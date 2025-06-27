import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re

@st.cache_resource
def load_models():
    ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
    model_name = "google/flan-t5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
    llm = HuggingFacePipeline(pipeline=pipe)
    return ner, llm

ner_pipeline, llm = load_models()

warn_template = """
You are a privacy assistant.

The user wrote:
"{input_text}"

This message contains the following personal information: {entities}

Write a response that includes:
- A short warning about the privacy risks
- One sentence about what could go wrong
"""

rewrite_template = """
You are a privacy assistant.

The user wrote:
"{input_text}"

You must remove or redact the following personal information: {entities}

Rules:
DO:
- Replace names, phone numbers, and locations with general phrases like "my friend", "my usual place", etc.
- Use natural, polite phrasing.
- Keep the sentence structure similar, but change every sensitive word.

DON’T:
- Repeat any personal information again
- Say "I cannot help" or refuse the task

Now rewrite the message below:
"""

warn_prompt = PromptTemplate(input_variables=["input_text", "entities"], template=warn_template)
rewrite_prompt = PromptTemplate(input_variables=["input_text", "entities"], template=rewrite_template)

warn_chain = LLMChain(llm=llm, prompt=warn_prompt)
rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt)

st.title("🔐 Privacy Guard – NLP Agent")
user_input = st.text_area("Enter a message to analyze for privacy leaks:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        ner_result = ner_pipeline(user_input)
        private_entities = [ent['word'] for ent in ner_result if ent['entity_group'] in ['PER', 'LOC', 'ORG', 'MISC']]
        manual_entities = re.findall(r'\b(?:\+31|0)[1-9][0-9]{7,8}\b', user_input)
        combined_entities = list(set(private_entities + manual_entities))

        warn_output = warn_chain.invoke({
            "input_text": user_input,
            "entities": ", ".join(combined_entities)
        })

        rewrite_output = rewrite_chain.invoke({
            "input_text": user_input,
            "entities": ", ".join(combined_entities)
        })

        st.subheader("🔎 Detected Entities")
        st.write(combined_entities or "No sensitive data detected.")

        st.subheader("⚠️ Warning & Risk Explanation")
        st.write(warn_output["text"] if isinstance(warn_output, dict) else warn_output)

        st.subheader("✅ Rewritten Message")
        st.write(rewrite_output["text"] if isinstance(rewrite_output, dict) else rewrite_output)
