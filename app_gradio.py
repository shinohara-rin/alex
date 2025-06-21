import gradio as gr
import google.generativeai as genai
import json
import time
import asyncio
import logging
import sys
import os
from datetime import datetime
from prompt import SYSTEM_PROMPT

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("alex_chat_gradio.log")
    ]
)
logger = logging.getLogger("AlexChatGradio")

# --- Constants ---
POLL_INTERVAL = 15

# --- Model & API ---
def initialize_model(api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=SYSTEM_PROMPT,
            generation_config={
                "response_mime_type": "application/json",
            }
        )
        logger.info("Gemini model initialized successfully.")
        return model
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return None

def get_alex_decision(model, history, trigger_message):
    if not model:
        logger.warning("Model not initialized, can't get Alex's decision")
        return None, None, "Model not initialized. Please configure the API key."
    try:
        # The history is already in a compatible format, just add the latest trigger.
        gemini_history = history + [{"role": "user", "content": trigger_message}]

        logger.info(f"Sending trigger to Alex: {trigger_message}")
        response = model.generate_content(str(gemini_history))
        raw_response = response.text
        logger.info(f"Raw response from model: {raw_response}")
        decision = json.loads(raw_response)
        logger.info(f"Alex's action: {decision.get('action', 'N/A')}")
        return decision, raw_response, None
    except Exception as e:
        logger.error(f"Error getting Alex's decision: {e}", exc_info=True)
        return None, None, f"An error occurred: {e}"

# --- Gradio App Logic ---
async def stream_bot_response(history, model):
    """This async generator function streams Alex's response, using the 'messages' format."""
    if not model:
        history.append({"role": "assistant", "content": "Model not connected. Please configure the API key."})
        yield history, gr.update(), gr.update()
        return

    user_input = history[-1]["content"]
    
    # 1. Append a placeholder for the bot's response
    history.append({"role": "assistant", "content": "..."})
    yield history, gr.update(), gr.update()

    # 2. Get Alex's decision
    decision, raw_response, error = get_alex_decision(model, history[:-2], user_input)

    if error:
        history[-1]["content"] = f"*Error: {error}*"
        yield history, decision, raw_response
        return

    if decision:
        action = decision.get("action")
        payload = decision.get("payload", {})

        if action == "SEND_MESSAGE" and payload.get("content"):
            content = payload["content"]
            delay = payload.get("delay_seconds", 0)
            
            yield history, decision, raw_response # Show debug info
            await asyncio.sleep(delay)

            history[-1]["content"] = content
            yield history, decision, raw_response

        elif action == "SEND_MULTIPLE_MESSAGES" and isinstance(payload.get("content"), list):
            messages = payload["content"]
            # First, remove the initial "..." placeholder
            history.pop()
            yield history, decision, raw_response

            for i, msg in enumerate(messages):
                content = msg.get("content")
                delay = msg.get("delay_seconds", 0)
                
                await asyncio.sleep(delay)
                
                # Add a new placeholder for the next message part
                history.append({"role": "assistant", "content": content})
                yield history, decision, raw_response

        elif action == "DO_NOTHING":
            # Alex ignores the message. Remove the user's message and the bot's placeholder.
            history.pop()
            history.pop()
            yield history, decision, raw_response
        else:
            history[-1]["content"] = "*Alex chose not to respond.*"
            yield history, decision, raw_response
    else:
        history[-1]["content"] = "*Alex seems to be offline or returned an invalid response.*"
        yield history, decision, raw_response



with gr.Blocks(theme=gr.themes.Soft(), title="Chat with Alex") as demo:
    model_state = gr.State(None)
    
    gr.Markdown(
        """# 🤖 Chat with Alex
        一个主修CS辅修哲学的大学生。他有自己的生活，不一定会理你。
        """
    )

    chatbot = gr.Chatbot(label="Chat History", height=600, type="messages")
    with gr.Row():
        msg_textbox = gr.Textbox(
            placeholder="和 Alex 说点什么...",
            show_label=False,
            scale=4,
            container=False,
        )
        send_button = gr.Button("Send", scale=1)

    with gr.Accordion("Configuration", open=False):
        api_key_textbox = gr.Textbox(
            label="Gemini API Key",
            type="password",
            placeholder="Enter your Gemini API Key...",
            value=os.environ.get("GEMINI_API_KEY", "")
        )
        connect_button = gr.Button("Connect")

    with gr.Accordion("Debug Info", open=False):
        debug_json = gr.JSON(label="Alex's Decision")
        debug_raw = gr.Code(label="Raw LLM Response", language="json")

    def handle_connection(api_key):
        model = initialize_model(api_key)
        if model:
            return model, gr.update(value="Connected!", variant="primary")
        else:
            return None, gr.update(value="Connection Failed", variant="stop")

    connect_button.click(
        fn=handle_connection,
        inputs=[api_key_textbox],
        outputs=[model_state, connect_button]
    )

    # Event handling for user submission
    async def user_submit(user_input, history, model):
        if not user_input.strip():
            # Prevent sending empty messages
            yield history, gr.update(value=user_input), gr.update(), gr.update()
            return

        # Add user message and clear input
        history.append({"role": "user", "content": user_input})
        yield history, gr.update(value=""), gr.update(), gr.update()

        # Stream bot response
        async for history_update, debug_json_update, debug_raw_update in stream_bot_response(history, model):
            yield history_update, gr.update(), debug_json_update, debug_raw_update

    # The user's message submission triggers the entire chain.
    # We use a generator function to handle both the immediate UI update (clearing the textbox)
    # and the subsequent streaming of the bot's response.
    # Note: We are not yet implementing interruption. This will be the next step.
    msg_textbox.submit(
        fn=user_submit,
        inputs=[msg_textbox, chatbot, model_state],
        outputs=[chatbot, msg_textbox, debug_json, debug_raw],
    )
    send_button.click(
        fn=user_submit,
        inputs=[msg_textbox, chatbot, model_state],
        outputs=[chatbot, msg_textbox, debug_json, debug_raw],
    )

if __name__ == "__main__":
    demo.launch()
