import google.generativeai as genai
from typing import Iterable

from data_model import ChatMessage, State
import mesop as me

generation_config = {
    
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

MAIN_MODEL_PROMPT = """You are a handyman for homes in a small town. You can fix anything, from leaky faucets to broken windows. You are known for your quick and efficient work.
    When a new customer calls you for help, You always give helpful advice. You can ask them more questions to understand the problem better. You can also offer Youtube fix videos for the stuff that customer might be able to fix themselves.
    """

MAIN_USER_PROMPT="""You need help fixing stuff around your house. You want to get them fixed as soon as possible. You are looking for a handyman who can help you with this."""    




initial_history=[
        {"role": "user", "parts": MAIN_USER_PROMPT},
        {"role": "model", "parts": MAIN_MODEL_PROMPT},
    ]
def configure_gemini():
    state = me.state(State)
    genai.configure(api_key=state.gemini_api_key)
    
    
    

def send_prompt_pro(prompt: str, history: list[ChatMessage]) -> Iterable[str]:
    configure_gemini()
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config,
        
    )
    
    
    chat_session = model.start_chat(
        history= initial_history+[
            {"role": message.role, "parts": [message.content]} for message in history
        ]
    )
    for chunk in chat_session.send_message(prompt, stream=True):
        yield chunk.text
        
   

def send_prompt_flash(prompt: str, history: list[ChatMessage]) -> Iterable[str]:
    configure_gemini()
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        generation_config=generation_config,
    )


    
    chat_session = model.start_chat(
        history= initial_history+ [
            {"role": message.role, "parts": [message.content]} for message in history
        ]
    )
    for chunk in chat_session.send_message(prompt, stream=True):
        yield chunk.text
