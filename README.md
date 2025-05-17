# NaturalLanguageToSQL
This project takes natural language as the input and outputs the corresponding SQL query for it. It uses Langchain/Langgraph and Generative AI to do this.

It uses both llama3-70b-8192 and gemini-2.0-flash to make this work. It invokes gemini only when the free llma3 model is not able to handle the query.

STEPS TO RUN:

1. Get both groq and gemini api keys and store them inside the .env file.
2. Install the latest version of python (I developed this with python 3.12).
3. Run the run.sh script OR install all the dependencies using `pip install -r requirements.txt` and then `py main.py`. It is recommended that you create a virtual environment and then activate before running these steps. If you run the script, this should be taken care for you.
4. Update your table_description in the `table_description.txt` to whatever your SQL schema is. Similarly, update your query in `question.txt` to whatver you want to ask this agent.

![Image of Graph](https://github.com/Chuckoo/NaturalLanguageToSQL/blob/main/mermaid.png "Mermaid Image")



