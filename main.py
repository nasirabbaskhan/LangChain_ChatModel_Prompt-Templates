from typing import Dict
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
from langchain.chains.sequential import SimpleSequentialChain
from langchain import  LLMChain, PromptTemplate
from dotenv import load_dotenv
import getpass
import os


# Set the environment variable for Google Cloud application credentials
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("GOOGLE_API_KEY")



# Initialize an instance of the ChatGoogleGenerativeAI with specific parameters
model =  ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Specify the model to use
    temperature=0.2,            # Set the randomness of the model's responses (0 = deterministic, 1 = very random)
)

# initializing the PromptTemplate
prompt1 = PromptTemplate(template="translate text into French:{text}", input_variables= ["text"] )
prompt2 = PromptTemplate(template="Wht is the sentiment of this French text {text}, please give only sentiment without detail", input_variables=["text"])

# initializing the LLMChain
chain1= LLMChain(llm=model, prompt= prompt1)
chain2= LLMChain(llm=model, prompt= prompt2)

# using the SimpleSequentialchain() to handle the given chains prompts
chain= SimpleSequentialChain(chains=[chain1, chain2])


#invoke the chain to get and print response
input_data = {"input": "I don not like cricket"}
response = chain.invoke(input_data)

print(response)


     