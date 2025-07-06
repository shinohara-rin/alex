# Alex - Human-like LLM Chat Agent
# Backend server implementation using Socket.IO and Google Gemini API

import socketio
import uvicorn
import os
import json
import asyncio
import time
import random
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Any
from prompt import SYSTEM_PROMPT

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

# Initialize Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = socketio.ASGIApp(sio)

# Agent state management
class AgentState:
    def __init__(self):
        self.mood = "neutral"  # neutral, happy, sad, excited, annoyed
        self.current_thoughts = "Just chilling, waiting for something interesting to happen."
        self.last_activity = "Browsing the web"
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.is_busy = False  # Whether Alex is "busy" with something else
        self.conversation_history = []  # Store recent messages
        self.max_history_length = 20  # Maximum number of messages to keep in memory
        
        # Message interruption tracking
        self.is_typing = False  # Whether Alex is currently typing
        self.typing_task = None  # Task reference for cancellation
        self.waiting_task = None  # Task reference for WAITING timer cancellation
        self.consecutive_interruptions = 0  # Count of consecutive interruptions
        self.max_consecutive_interruptions = 5  # Max interruptions before cooldown
        self.cooldown_active = False  # Whether cooldown is active
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mood": self.mood,
            "current_thoughts": self.current_thoughts,
            "last_activity": self.last_activity,
            "updated_at": self.updated_at,
            "is_busy": self.is_busy,
            "is_typing": self.is_typing,
            "consecutive_interruptions": self.consecutive_interruptions,
            "cooldown_active": self.cooldown_active
        }
    
    def update_state(self, new_state: Dict[str, Any]) -> None:
        """Update agent state with new values"""
        if "mood" in new_state:
            self.mood = new_state["mood"]
        if "current_thoughts" in new_state:
            self.current_thoughts = new_state["current_thoughts"]
        if "last_activity" in new_state:
            self.last_activity = new_state["last_activity"]
        if "is_busy" in new_state:
            self.is_busy = new_state["is_busy"]
        
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def add_message_to_history(self, message: Dict[str, Any]) -> None:
        """Add a message to conversation history"""
        self.conversation_history.append(message)
        # Keep history limited to max length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def get_state_for_debug(self) -> Dict[str, Any]:
        """Get the current state of the agent for debugging"""
        return {
            "mood": self.mood,
            "current_thoughts": self.current_thoughts,
            "last_activity": self.last_activity,
            "is_busy": self.is_busy,
            "is_typing": self.is_typing,
            "consecutive_interruptions": self.consecutive_interruptions,
            "cooldown_active": self.cooldown_active,
            "conversation_history_length": len(self.conversation_history)
        }

# Initialize agent state
agent = AgentState()

# Helper function to construct prompt with agent state and conversation history
def construct_prompt(user_message: Optional[str] = None, is_system_message: bool = False) -> str:
    # Format conversation history
    history_text = ""
    for msg in agent.conversation_history:
        sender = "User" if msg["sender"] == "user" else "Alex"
        history_text += f"{sender}: {msg['text']}\n"
    
    # Construct context with agent state
    context = f"""
Current agent state:
- Mood: {agent.mood}
- Current thoughts: {agent.current_thoughts}
- Last activity: {agent.last_activity}
- Is busy: {'Yes' if agent.is_busy else 'No'}
- Last updated: {agent.updated_at}

Recent conversation:
{history_text}
"""
    
    # Add user message if provided
    if user_message:
        if is_system_message:
            # For system messages like timeout notifications
            prompt = f"{context}\n\n{user_message}\n\nHow do you respond as Alex?"
        else:
            # For regular user messages
            prompt = f"{context}\n\nUser just sent: \"{user_message}\"\n\nHow do you respond as Alex?"
    else:
        # For heartbeat/autonomous messages
        time_since_last_message = "a while" if not agent.conversation_history else "recently"
        prompt = f"{context}\n\nIt's been {time_since_last_message} since the last message. Do you want to initiate a conversation or share a thought?"
    
    return prompt

# Parse LLM response to ensure it's valid JSON
async def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM response and ensure it's valid JSON"""
    try:
        # Extract JSON from response if needed (in case model outputs additional text)
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            response_data = json.loads(json_str)
            
            # Validate required fields
            if "action" not in response_data:
                raise ValueError("Missing 'action' field in response")
            
            return response_data
        else:
            print(f"No valid JSON found in response: {response_text}")
            raise ValueError("No valid JSON found in response")
    except json.JSONDecodeError:
        # Fallback for invalid JSON
        print(f"Error parsing LLM response as JSON: {response_text}")
        return {
            "thought": "Error parsing response",
            "action": "SEND_MESSAGE",
            "payload": {
                "content": "Sorry, I got distracted. What were we talking about?",
                "delay_seconds": 1
            }
        }

# Process agent response and take appropriate action
async def process_agent_response(response_data: Dict[str, Any], sid: str, raw_llm_response: str = None, prompt: str = None) -> None:
    """Process the agent's response and perform the appropriate action"""
    # Extract agent thoughts for potential state updates
    thought = response_data.get("thought", "")
    action = response_data.get("action", "DO_NOTHING")
    payload = response_data.get("payload", {})
    
    # Update agent state based on thoughts (simple heuristic)
    new_state = {}
    
    # Simple mood detection from thought
    if "happy" in thought.lower() or "excited" in thought.lower() or "interesting" in thought.lower():
        new_state["mood"] = "happy"
    elif "sad" in thought.lower() or "depressed" in thought.lower() or "upset" in thought.lower():
        new_state["mood"] = "sad"
    elif "annoyed" in thought.lower() or "irritated" in thought.lower() or "angry" in thought.lower():
        new_state["mood"] = "annoyed"
    
    # Update current thoughts
    if thought:
        new_state["current_thoughts"] = thought[:100]  # Truncate if too long
    
    # Update agent state
    agent.update_state(new_state)
    
    # Send agent status update to client
    await sio.emit("agent_status", {
        "mood": agent.mood,
        "activity": agent.current_thoughts
    }, room=sid)
    
    # Send debug information to client
    await sio.emit("debug_info", {
        "timestamp": time.time(),
        "agent_state": agent.get_state_for_debug(),
        "raw_llm_response": raw_llm_response,
        "parsed_response": response_data,
        "conversation_history": agent.conversation_history,
        "llm_prompt": prompt
    }, room=sid)
    
    # Perform action based on response
    if action == "SEND_MESSAGE":
        content = payload.get("content", "")
        delay_seconds = payload.get("delay_seconds", 0)
        
        if content:
            # Create a function to handle the message sending with delay
            async def send_delayed_message(content, delay_seconds):
                try:
                    # Mark agent as typing
                    agent.is_typing = True
                    
                    # Show typing indicator
                    await sio.emit("agent_typing", {}, room=sid)
                    
                    # Simulate typing delay
                    if delay_seconds > 0:
                        await asyncio.sleep(delay_seconds)
                    else:
                        # Ensure minimum typing delay for UX
                        typing_delay = max(0.5, len(content) * 0.05)  # ~50ms per character with minimum
                        await asyncio.sleep(typing_delay)
                    
                    # Send message to user
                    message_data = {"text": content, "sender": "agent", "timestamp": time.time()}
                    await sio.emit("message", message_data, room=sid)
                    
                    # Add to conversation history
                    agent.add_message_to_history(message_data)
                    
                    # Reset consecutive interruptions counter on successful message
                    agent.consecutive_interruptions = 0
                    
                except asyncio.CancelledError:
                    # Message was cancelled due to user interruption
                    interrupted_message = {"text": f"[Interrupted while typing: {content}]", 
                                          "sender": "agent", 
                                          "timestamp": time.time(),
                                          "was_interrupted": True}
                    
                    # Add to conversation history but don't send to user
                    agent.add_message_to_history(interrupted_message)
                    
                    # Increment consecutive interruptions counter
                    agent.consecutive_interruptions += 1
                    
                    # Check if we need to activate cooldown
                    if agent.consecutive_interruptions >= agent.max_consecutive_interruptions:
                        agent.cooldown_active = True
                        
                        # Create a task to reset cooldown after some time
                        async def reset_cooldown():
                            await asyncio.sleep(30)  # 30 seconds cooldown
                            agent.cooldown_active = False
                            agent.consecutive_interruptions = 0
                        
                        asyncio.create_task(reset_cooldown())
                finally:
                    # Reset typing state
                    agent.is_typing = False
                    agent.typing_task = None
            
            # Cancel any existing typing task
            if agent.typing_task and not agent.typing_task.done():
                agent.typing_task.cancel()
            
            # Create and store the new typing task
            agent.typing_task = asyncio.create_task(send_delayed_message(content, delay_seconds))
    
    elif action == "SEND_MULTIPLE_MESSAGES":
        messages = payload.get("content", [])
        
        # Create a function to handle sending multiple messages with delays
        async def send_multiple_delayed_messages(messages):
            try:
                for i, msg in enumerate(messages):
                    content = msg.get("content", "")
                    delay_seconds = msg.get("delay_seconds", 0)
                    
                    if content:
                        try:
                            # Mark agent as typing
                            agent.is_typing = True
                            
                            # Show typing indicator for each message
                            await sio.emit("agent_typing", {}, room=sid)
                            
                            # Simulate typing delay
                            if delay_seconds > 0:
                                await asyncio.sleep(delay_seconds)
                            else:
                                # Ensure minimum typing delay for UX
                                typing_delay = max(0.5, len(content) * 0.05)  # ~50ms per character with minimum
                                await asyncio.sleep(typing_delay)
                            
                            # Send message to user
                            message_data = {"text": content, "sender": "agent", "timestamp": time.time()}
                            await sio.emit("message", message_data, room=sid)
                            
                            # Add to conversation history
                            agent.add_message_to_history(message_data)
                            
                            # If not the last message, add a small pause between messages
                            if i < len(messages) - 1:
                                await asyncio.sleep(0.5)  # Small pause between consecutive messages
                                
                        except asyncio.CancelledError:
                            # This specific message was cancelled
                            interrupted_message = {"text": f"[Interrupted while typing: {content}]", 
                                                 "sender": "agent", 
                                                 "timestamp": time.time(),
                                                 "was_interrupted": True}
                            
                            # Add to conversation history but don't send to user
                            agent.add_message_to_history(interrupted_message)
                            raise  # Re-raise to cancel the entire sequence
                
                # Reset consecutive interruptions counter on successful completion of all messages
                agent.consecutive_interruptions = 0
                
            except asyncio.CancelledError:
                # Entire message sequence was cancelled
                # Increment consecutive interruptions counter
                agent.consecutive_interruptions += 1
                
                # Check if we need to activate cooldown
                if agent.consecutive_interruptions >= agent.max_consecutive_interruptions:
                    agent.cooldown_active = True
                    
                    # Create a task to reset cooldown after some time
                    async def reset_cooldown():
                        await asyncio.sleep(30)  # 30 seconds cooldown
                        agent.cooldown_active = False
                        agent.consecutive_interruptions = 0
                    
                    asyncio.create_task(reset_cooldown())
            finally:
                # Reset typing state
                agent.is_typing = False
                agent.typing_task = None
        
        # Cancel any existing typing task
        if agent.typing_task and not agent.typing_task.done():
            agent.typing_task.cancel()
        
        # Create and store the new typing task
        agent.typing_task = asyncio.create_task(send_multiple_delayed_messages(messages))
    
    # Handle IGNORE action (similar to previous DO_NOTHING)
    elif action == "IGNORE":
        # No action needed, agent chooses to ignore the message
        pass
    
    # Handle WAITING action - start a 10-second timer
    elif action == "WAITING":
        # Create a task to wait for 10 seconds and then check if user sent a message
        async def waiting_timeout(sid, user_message_text):
            try:
                # Wait for 10 seconds
                await asyncio.sleep(10)
                
                # Check if we received any new messages from the user in the meantime
                # We'll compare the last message in history with the one that triggered this waiting
                if agent.conversation_history and agent.conversation_history[-1]["sender"] == "user":
                    last_user_message = agent.conversation_history[-1]["text"]
                    if last_user_message == user_message_text:
                        # No new message received, inform the LLM and ask for a new decision
                        prompt = construct_prompt(
                            "[SYSTEM: User did not continue their message after 10 seconds. Please decide what to do now.]", 
                            is_system_message=True
                        )
                        
                        # Generate response with correct roles for Gemini API
                        response = await model.generate_content_async([
                            {"role": "user", "parts": [SYSTEM_PROMPT]},
                            {"role": "user", "parts": [prompt]}
                        ])
                        response_text = response.text
                        
                        # Parse the response - properly await the coroutine
                        parsed_response = await parse_llm_response(response_text)
                        
                        # Process the new response
                        await process_agent_response(parsed_response, sid, response_text, prompt)
            except asyncio.CancelledError:
                # WAITING task was cancelled due to user interruption
                interrupted_message = {"text": "[WAITING timer interrupted due to new user message]", 
                                      "sender": "system", 
                                      "timestamp": time.time(),
                                      "was_interrupted": True}
                
                # Add to conversation history but don't send to user
                agent.add_message_to_history(interrupted_message)
            finally:
                # Clear the waiting task reference
                agent.waiting_task = None
            
        # Start the waiting timeout task
        last_user_message = agent.conversation_history[-1]["text"] if agent.conversation_history and agent.conversation_history[-1]["sender"] == "user" else ""
        # Cancel any existing waiting task
        if agent.waiting_task and not agent.waiting_task.done():
            agent.waiting_task.cancel()
        # Create and store the new waiting task
        agent.waiting_task = asyncio.create_task(waiting_timeout(sid, last_user_message))

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    print(f'User connected: {sid}')
    
    # Send initial agent status to the newly connected client
    await sio.emit("agent_status", {
        "mood": agent.mood,
        "activity": agent.current_thoughts or "Just chilling"
    }, room=sid)
    
    # Start heartbeat task if it's not already running
    for task in asyncio.all_tasks():
        if task.get_name() == 'agent_heartbeat':
            break
    else:
        # No heartbeat task found, start one
        task = asyncio.create_task(agent_heartbeat())
        task.set_name('agent_heartbeat')

@sio.event
async def message(sid, data):
    print(f'Message received: {data}')
    
    # Add user message to conversation history
    user_message = {
        "text": data.get("text", ""),
        "sender": "user",
        "timestamp": time.time()
    }
    agent.add_message_to_history(user_message)
    
    # Debug commands for testing
    if user_message["text"].strip().lower() == "!debug cooldown":
        agent.consecutive_interruptions = agent.max_consecutive_interruptions
        agent.cooldown_active = True
        debug_message = {
            "text": "[DEBUG: Cooldown mode activated for testing]",
            "sender": "system",
            "timestamp": time.time()
        }
        await sio.emit("message", debug_message, room=sid)
        agent.add_message_to_history(debug_message)
        
        # Create a task to reset cooldown after some time
        async def reset_cooldown():
            await asyncio.sleep(30)  # 30 seconds cooldown
            agent.cooldown_active = False
            agent.consecutive_interruptions = 0
            reset_message = {
                "text": "[DEBUG: Cooldown mode deactivated]",
                "sender": "system",
                "timestamp": time.time()
            }
            await sio.emit("message", reset_message, room=sid)
            agent.add_message_to_history(reset_message)
        
        asyncio.create_task(reset_cooldown())
        return
    
    # Debug command to test interruption
    elif user_message["text"].strip().lower() == "!debug typing":
        # Create a long typing delay that can be interrupted
        async def simulate_long_typing():
            try:
                # Mark agent as typing
                agent.is_typing = True
                
                # Show typing indicator
                await sio.emit("agent_typing", {}, room=sid)
                
                # Simulate a long typing delay (10 seconds)
                await asyncio.sleep(10)
                
                # If not interrupted, send the message
                message_data = {
                    "text": "This is a test message that took a long time to type. If you see this, it means you didn't interrupt me while I was typing!", 
                    "sender": "agent", 
                    "timestamp": time.time()
                }
                await sio.emit("message", message_data, room=sid)
                agent.add_message_to_history(message_data)
                
            except asyncio.CancelledError:
                # Message was cancelled due to user interruption
                interrupted_message = {
                    "text": "[DEBUG: Typing was interrupted!]", 
                    "sender": "system", 
                    "timestamp": time.time()
                }
                
                # Add to conversation history but don't send to user
                agent.add_message_to_history(interrupted_message)
                await sio.emit("message", interrupted_message, room=sid)
                
                # Increment consecutive interruptions counter
                agent.consecutive_interruptions += 1
                
            finally:
                # Reset typing state
                agent.is_typing = False
                agent.typing_task = None
        
        # Cancel any existing typing task
        if agent.typing_task and not agent.typing_task.done():
            agent.typing_task.cancel()
        
        # Create and store the new typing task
        agent.typing_task = asyncio.create_task(simulate_long_typing())
        
        debug_message = {
            "text": "[DEBUG: Started a 10-second typing simulation. Send any message to interrupt it.]",
            "sender": "system",
            "timestamp": time.time()
        }
        await sio.emit("message", debug_message, room=sid)
        agent.add_message_to_history(debug_message)
        return
    
    # Check if Alex is currently typing or waiting - if so, cancel the tasks
    if agent.typing_task and not agent.typing_task.done():
        print(f"Interrupting Alex's typing due to new user message")
        agent.typing_task.cancel()
        
    # Check if there's a pending WAITING timer - if so, cancel it
    if agent.waiting_task and not agent.waiting_task.done():
        print(f"Cancelling Alex's WAITING timer due to new user message")
        agent.waiting_task.cancel()
    
    # Check if we're in cooldown mode due to too many interruptions
    if agent.cooldown_active:
        # Send a message to the user about the cooldown
        cooldown_message = {
            "text": "[Alex is taking a moment to gather thoughts. Please wait a moment before sending more messages.]", 
            "sender": "system",
            "timestamp": time.time()
        }
        await sio.emit("message", cooldown_message, room=sid)
        agent.add_message_to_history(cooldown_message)
        return  # Don't process this message with the LLM
    
    # Construct prompt with agent state and user message
    prompt = construct_prompt(user_message["text"])
    
    # Generate response with correct roles for Gemini API
    response = await model.generate_content_async([
        {"role": "user", "parts": [SYSTEM_PROMPT]},
        {"role": "user", "parts": [prompt]}
    ])
    
    # Save raw LLM response for debugging
    raw_response = response.text
    
    # Parse and process response
    response_data = await parse_llm_response(raw_response)
    await process_agent_response(response_data, sid, raw_response, prompt)

@sio.event
async def disconnect(sid):
    print(f'User disconnected: {sid}')

# Heartbeat loop for autonomous agent behavior
async def agent_heartbeat():
    """Periodic task that allows the agent to act autonomously"""
    heartbeat_interval = 60 * 15  # 15 minutes in seconds
    min_interval = 60 * 5  # Minimum 5 minutes between autonomous messages
    
    last_autonomous_message = time.time()
    
    while True:
        await asyncio.sleep(heartbeat_interval)
        
        # Only proceed if enough time has passed since last autonomous message
        if (time.time() - last_autonomous_message) >= min_interval:
            # Random chance to initiate conversation (30%)
            if random.random() < 0.3:
                # Construct prompt for autonomous message
                prompt = construct_prompt()
                
                # Generate autonomous message
                response = await model.generate_content_async([
                    {"role": "user", "parts": [SYSTEM_PROMPT]},
                    {"role": "user", "parts": [prompt]}
                ])
                
                # Parse and process response
                response_data = await parse_llm_response(response.text)
                
                # Only send if action is to send a message
                if response_data.get("action") in ["SEND_MESSAGE", "SEND_MULTIPLE_MESSAGES"]:
                    # In single-user mode, broadcast to all clients
                    await sio.emit("message", {
                        "text": "[Autonomous message suppressed in debug mode]",
                        "sender": "system",
                        "timestamp": time.time()
                    })
                    
                    # Update last message timestamp
                    last_autonomous_message = time.time()
                    
                    # Update last activity
                    agent.update_state({"last_activity": "Sending a spontaneous message"})

# Start the server
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)
