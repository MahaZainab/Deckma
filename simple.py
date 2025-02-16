import requests

# Replace with your DeepSeek R1 API endpoint and key
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"  # Example endpoint
API_KEY = "sk-27a1ba1ece2c436e974559bca9878bfc"

def send_to_deepseek(user_input, context=None):
    """
    Send user input to the DeepSeek R1 API and get the response.
    :param user_input: The user's message.
    :param context: Optional context for multi-turn conversations.
    :return: The chatbot's response.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "message": user_input,
        "context": context or []  # Include context if available
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json().get("response", "Sorry, I couldn't process that.")
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def chatbot():
    """Simple chatbot loop."""
    print("Hello! I'm your chatbot powered by DeepSeek R1. Type 'exit' to end the conversation.")
    context = []  # Store conversation context for multi-turn interactions
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        # Get response from DeepSeek R1
        bot_response = send_to_deepseek(user_input, context)
        print(f"Chatbot: {bot_response}")
        # Update context for multi-turn conversations (if supported by the API)
        context.append({"role": "user", "content": user_input})
        context.append({"role": "assistant", "content": bot_response})

if __name__ == "__main__":
    chatbot()