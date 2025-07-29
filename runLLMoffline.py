from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All


# model weights path
#(download models for gpt4all)
PATH='./nous-hermes-13b.ggmlv3.q4_0.bin'


# Create LLM Class
llm = GPT4All(model=PATH, verbose=True)


# Create a prompt template
prompt = PromptTemplate(
input_variables=['instruction', 'input', 'response'],
template="""
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
""" )


#llm chain
llmchain = LLMChain(prompt=prompt, llm=llm)


# Run the prompt
llmchain.run(
instruction="""Think and explain your answer""",
input="""Q: What is your name ?""",
response='A: ')
