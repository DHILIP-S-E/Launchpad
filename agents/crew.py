from crewai import Crew, Process
from agents import researcher, writer
from tasks import researchtask, writertask

# Create the crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[researchtask, writertask],
    process=Process.sequential,  # Sequential execution
)

# Kick off the crew
outcome = crew.kickoff(inputs={'topic': 'Spiking Neural Networks'})
print(outcome)
