from crewai import Agent
from langchain_groq import ChatGroq
from tools import tool

# Initialize the language model
groqllm = ChatGroq(model='llama3-8b-8192', temperature=0, verbose=True)

# Define the researcher agent
researcher = Agent(
    role="Experienced Researcher",
    goal='Uncover ground breaking technologies in {topic}',
    memory=True,
    llm=groqllm,
    backstory=("Driven by curiosity, you're at the forefront of innovation, eager to explore and share knowledge that could change the world."),
    tools=[tool],
    allow_delegation=True
)

# Define the writer agent
writer = Agent(
    role="Article Writer",
    goal='Narrate compelling tech stories about {topic}',
    backstory=("With a flair for simplifying complex topics, you craft engaging narratives that captivate and educate, bringing new discoveries to light in an accessible manner."),
    tools=[tool],
    verbose=True,
    llm=groqllm,
    memory=True,
    allow_delegation=False
)
