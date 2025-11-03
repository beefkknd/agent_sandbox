"""A simple "Hello World" example for the Gemini 2.5 Flash model."""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from google.auth.exceptions import DefaultCredentialsError


def hello_gemini():
    """
    Initializes the Gemini 1.5 Flash model, sends a prompt, and prints the response.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Get the API key from the environment variable
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        print("GEMINI_API_KEY not found in .env file or environment variables.")
        return

    try:
        # Pass the API key directly to the constructor
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0,
            google_api_key=gemini_api_key,
        )

        prompt = "who are you"
        print(f"Prompt: {prompt}")

        # Invoke the model with a HumanMessage
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])

        print("Response:")
        print(response.content)
    except DefaultCredentialsError:
        print("Authentication Error: Your default credentials were not found.")
        print("Please set up Application Default Credentials by following the instructions here:")
        print("https://cloud.google.com/docs/authentication/external/set-up-adc")
        print("Alternatively, you can set the GOOGLE_API_KEY environment variable.")


if __name__ == "__main__":
    hello_gemini()
