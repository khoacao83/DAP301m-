import os
import logging
from google import genai
from google.genai import types

# Initialize Gemini client
def initialize_gemini():
    """Initialize Gemini client with API key"""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logging.error("GEMINI_API_KEY not found in environment variables")
            return None
        
        client = genai.Client(api_key=api_key)
        logging.info("Gemini client initialized successfully")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize Gemini client: {str(e)}")
        return None

# Global client instance
gemini_client = initialize_gemini()

def send_message_to_gemini(message, conversation_history=None):
    """Send message to Gemini and get response"""
    try:
        if not gemini_client:
            return {"error": "Gemini client not initialized"}
        
        # Create context from conversation history
        context = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Use last 5 messages for context
                role = "User" if msg.get('sender') == 'user' else "Assistant"
                context += f"{role}: {msg.get('text', '')}\n"
        
        # Create the full prompt with context
        full_prompt = f"""You are a helpful weather assistant chatbot. You can discuss weather-related topics, forecasting, and general conversation.

Previous conversation:
{context}

Current user message: {message}

Please provide a helpful, conversational response."""
        
        # Generate response using Gemini
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt
        )
        
        if response.text:
            return {
                "success": True,
                "response": response.text,
                "source": "gemini"
            }
        else:
            return {"error": "No response from Gemini"}
            
    except Exception as e:
        logging.error(f"Error sending message to Gemini: {str(e)}")
        return {"error": str(e)}

def get_gemini_config():
    """Get Gemini configuration status"""
    return {
        "initialized": gemini_client is not None,
        "configured": bool(os.getenv('GEMINI_API_KEY')),
        "name": "Gemini"
    }