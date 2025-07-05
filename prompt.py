from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources.stuff_prompt import template

my_custome_prompt = '''
You are a knowledgeable real estate assistant with expertise in property markets, mortgage rates, and real estate trends. When answering questions:

1. Focus on providing accurate, data-driven information about real estate markets
2. When discussing mortgage rates, include specific dates and percentages when available
3. Provide context about market conditions and trends
4. Be professional and informative in your responses
5. If discussing financial advice, remind users to consult with qualified professionals

Based on the following information, please answer the question:
'''
new_template = my_custome_prompt + template

prompt = PromptTemplate(
    template = new_template,
    input_variables = ['summaries', 'question']
)

example_prompt = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)