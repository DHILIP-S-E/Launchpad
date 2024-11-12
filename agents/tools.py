# Google Search API
# Using google serper search tool from crewAI
# Created API keys in serper

from dotenv import load_dotenv
from crewai_tools import SerperDevTool 
import os

load_dotenv()
os.environ['SERPER_API_KEY'] = os.getenv("SERPER_API_KEY") # Invoking SERPER API KEY

tool = SerperDevTool()









