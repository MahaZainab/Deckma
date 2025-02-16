import requests

class DeepSeekChatbot:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key

    def get_response(self, user_input):
        """
        Send user input to the DeepSeek API and return the model's response.
        """
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'prompt': user_input,
            'max_tokens': 150,
            'temperature': 0.7  # Adjust based on how deterministic you want the responses
        }
        response = requests.post(self.api_url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json().get('choices')[0].get('text').strip()
        else:
            return "I'm sorry, I couldn't fetch a response. Please try again."

if __name__ == "__main__":
    # Replace 'your_api_url_here' and 'your_api_key_here' with your actual DeepSeek API URL and key.
    api_url = 'https://api.deepseek.com/call'  # Example, replace with actual
    api_key = 'sk-27a1ba1ece2c436e974559bca9878bfc'
    chatbot = DeepSeekChatbot(api_url, api_key)

    print("Hello! I'm here to help you. You can start chatting with me now. (type 'quit' to stop)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        response = chatbot.get_response(user_input)
        print("Bot:", response)
