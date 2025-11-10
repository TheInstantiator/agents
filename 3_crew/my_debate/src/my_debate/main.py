#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from my_debate.crew import Debate

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew.
    """
    inputs = {
        'motion': 'It is better to go for the touchdown on the oppenents 1 yard line on fourth down then to go for the field goal in American football.',
    }
    
    try:
        result = Debate().crew().kickoff(inputs=inputs)
        print(result.raw)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
