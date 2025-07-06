import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import './App.css';

// Debug panel component to display agent state and debug info
const DebugPanel = ({ debugData, showDebug, onClose }) => {
  if (!showDebug) return null;
  
  const formatTime = (timestamp) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp * 1000).toLocaleTimeString();
  };
  
  return (
    <div className="debug-panel">
      <div className="debug-header">
        <h3>Debug Panel</h3>
        <button className="close-button" onClick={onClose}>×</button>
      </div>
      
      <div className="debug-sections">
        <div className="debug-section">
          <h4>Agent State</h4>
          {debugData.agent_state ? (
            <div className="agent-state-ui">
              <div className="agent-state-row">
                <span className="state-label">Mood:</span>
                <div className="state-value mood-selector">
                  <span className={`mood-indicator ${debugData.agent_state.mood || 'neutral'}`}>
                    {debugData.agent_state.mood || 'neutral'}
                  </span>
                </div>
              </div>
              
              <div className="agent-state-row">
                <span className="state-label">Activity:</span>
                <div className="state-value">
                  <input 
                    type="text" 
                    readOnly 
                    value={debugData.agent_state.current_thoughts || 'Idle'}
                    className="activity-display"
                  />
                </div>
              </div>
              
              <div className="agent-state-row">
                <span className="state-label">Connected Users:</span>
                <div className="state-value">
                  <span className="user-count">
                    {debugData.agent_state.connected_users?.length || 0} users
                  </span>
                </div>
              </div>
              
              <div className="agent-state-row">
                <span className="state-label">Raw State:</span>
                <div className="state-value">
                  <details>
                    <summary>View raw state</summary>
                    <pre className="raw-state">
                      {JSON.stringify(debugData.agent_state, null, 2)}
                    </pre>
                  </details>
                </div>
              </div>
            </div>
          ) : (
            <p>No agent state data available</p>
          )}
        </div>
        
        <div className="debug-section">
          <h4>Latest Raw LLM Context</h4>
          {debugData.llm_prompt ? (
            <pre className="raw-llm-prompt">
              {debugData.llm_prompt}
            </pre>
          ) : (
            <p>No LLM prompt data available</p>
          )}
        </div>
        
        <div className="debug-section">
          <h4>Latest Raw LLM Response</h4>
          {debugData.raw_llm_response ? (
            <pre className="raw-llm">
              {debugData.raw_llm_response}
            </pre>
          ) : (
            <p>No raw LLM response data available</p>
          )}
        </div>
        
        <div className="debug-section">
          <h4>Parsed Response</h4>
          {debugData.parsed_response ? (
            <pre>
              {JSON.stringify(debugData.parsed_response, null, 2)}
            </pre>
          ) : (
            <p>No parsed response data available</p>
          )}
        </div>
        
        <div className="debug-section">
          <h4>Conversation History</h4>
          {debugData.conversation_history && debugData.conversation_history.length > 0 ? (
            <div className="debug-conversation">
              {debugData.conversation_history.map((msg, idx) => (
                <div key={idx} className={`debug-message ${msg.sender}`}>
                  <div className="debug-message-meta">
                    <span className="debug-message-sender">{msg.sender}</span>
                    <span className="debug-message-time">{formatTime(msg.timestamp)}</span>
                  </div>
                  <div className="debug-message-text">{msg.text}</div>
                </div>
              ))}
            </div>
          ) : (
            <p>No conversation history available</p>
          )}
        </div>
      </div>
    </div>
  );
};

// Default backend URL
const DEFAULT_BACKEND_URL = 'http://localhost:8080';

function App() {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [agentStatus, setAgentStatus] = useState({ mood: 'neutral', activity: 'Thinking...' });
  const [backendUrl, setBackendUrl] = useState(
    localStorage.getItem('backendUrl') || DEFAULT_BACKEND_URL
  );
  const [tempBackendUrl, setTempBackendUrl] = useState(backendUrl);
  const [showSettings, setShowSettings] = useState(false);
  const [debugData, setDebugData] = useState({
    agent_state: null,
    raw_llm_response: null,
    parsed_response: null,
    conversation_history: []
  });
  const [showDebug, setShowDebug] = useState(false);
  const chatWindowRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Function to connect to backend with specified URL
  const connectToBackend = (url) => {
    // Disconnect current socket if connected
    if (socket) {
      socket.disconnect();
    }
    
    // Create new socket with the URL
    socket = io(url);
    
    // Save URL to localStorage
    localStorage.setItem('backendUrl', url);
    setBackendUrl(url);
    
    // Reset connection state
    setIsConnected(false);
    
    // Setup event listeners for the new socket
    setupSocketListeners();
  };
  
  // Function to setup socket event listeners
  const setupSocketListeners = () => {
    socket.on('connect', () => {
      console.log('Connected to server');
      setIsConnected(true);
    });

    socket.on('disconnect', () => {
      setIsConnected(false);
    });

    socket.on('message', (message) => {
      // Clear typing indicator when message arrives
      setIsTyping(false);
      
      // Add timestamp if not provided
      if (!message.timestamp) {
        message.timestamp = Date.now();
      }
      
      setMessages((prevMessages) => [...prevMessages, message]);
    });
    
    socket.on('agent_typing', () => {
      setIsTyping(true);
    });
    
    socket.on('agent_status', (status) => {
      setAgentStatus(status);
    });
    
    // Listen for debug information
    socket.on('debug_info', (data) => {
      console.log('Debug info received:', data);
      setDebugData(data);
    });
  };
  
  // Handle settings form submission
  const handleSettingsSubmit = (e) => {
    e.preventDefault();
    connectToBackend(tempBackendUrl);
    setShowSettings(false);
  };

  useEffect(() => {
    console.log('Connecting to backend at:', backendUrl);
    
    // Initialize socket connection
    const newSocket = io(backendUrl, { 
      transports: ['websocket', 'polling'],
      reconnectionAttempts: 5,
      reconnectionDelay: 1000
    });
    
    // Set up event listeners
    newSocket.on('connect', () => {
      console.log('Connected to backend');
      setIsConnected(true);
    });
    
    newSocket.on('connect_error', (err) => {
      console.error('Connection error:', err);
      setIsConnected(false);
    });
    
    newSocket.on('disconnect', () => {
      console.log('Disconnected from backend');
      setIsConnected(false);
    });
    
    newSocket.on('agent_typing', () => {
      console.log('Agent is typing...');
      setIsTyping(true);
    });
    
    newSocket.on('message', (data) => {
      console.log('Message received:', data);
      setIsTyping(false);
      
      // Add timestamp if not provided
      if (!data.timestamp) {
        data.timestamp = Date.now();
      }
      
      setMessages(prevMessages => [...prevMessages, data]);
    });
    
    newSocket.on('agent_status', (data) => {
      console.log('Agent status update:', data);
      setAgentStatus({
        mood: data.mood || 'neutral',
        activity: data.activity || 'Idle'
      });
    });
    
    // Listen for debug information
    newSocket.on('debug_info', (data) => {
      console.log('Debug info received:', data);
      setDebugData(data);
    });
    
    // Store socket instance
    setSocket(newSocket);
    
    // Clean up on unmount
    return () => {
      console.log('Cleaning up socket connection');
      if (newSocket) {
        newSocket.disconnect();
      }
    };
  }, [backendUrl]);

  const sendMessage = () => {
    if (inputMessage.trim() && socket) {
      const message = { 
        text: inputMessage, 
        sender: 'user',
        timestamp: Date.now()
      };
      socket.emit('message', message);
      setMessages((prevMessages) => [...prevMessages, message]);
      setInputMessage('');
    }
  };
  
  // Format timestamp to readable time
  const formatTime = (timestamp) => {
    if (!timestamp) return '';
    
    // Check if timestamp is already in milliseconds (JavaScript timestamp) or seconds (Python timestamp)
    const date = timestamp > 1000000000000 ? new Date(timestamp) : new Date(timestamp * 1000);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="App">
      <header className="chat-header">
        <div className="header-main">
          <h1>Alex</h1>
          <div className="header-buttons">
            <button 
              className="debug-toggle-button" 
              onClick={() => setShowDebug(!showDebug)}
              title="Debug Panel"
            >
              🐞
            </button>
            <button 
              className="settings-button" 
              onClick={() => setShowSettings(!showSettings)}
              title="Settings"
            >
              ⚙️
            </button>
          </div>
        </div>
        <div className="status-indicator">
          <span className={`status-dot ${isConnected ? 'online' : 'offline'}`}></span>
          <span>{isConnected ? 'Online' : 'Offline'}</span>
          <span className="backend-url">({backendUrl})</span>
        </div>
        <div className="agent-status">
          <div className="mood">Mood: {agentStatus.mood}</div>
          <div className="activity">{agentStatus.activity}</div>
        </div>
      </header>
      
      {showSettings && (
        <div className="settings-panel">
          <form onSubmit={handleSettingsSubmit}>
            <h3>Backend Settings</h3>
            <div className="settings-field">
              <label htmlFor="backendUrl">Backend URL:</label>
              <input
                id="backendUrl"
                type="text"
                value={tempBackendUrl}
                onChange={(e) => setTempBackendUrl(e.target.value)}
                placeholder="http://localhost:8000"
              />
            </div>
            <div className="settings-actions">
              <button type="submit">Connect</button>
              <button type="button" onClick={() => setShowSettings(false)}>Cancel</button>
            </div>
          </form>
        </div>
      )}
      
      {/* Debug Panel */}
      <DebugPanel 
        debugData={debugData} 
        showDebug={showDebug} 
        onClose={() => setShowDebug(false)} 
      />
      
      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`message-container ${msg.sender}`}>
            <div className="message-content">
              <div className="message">
                {msg.text}
              </div>
              <div className="message-time">{formatTime(msg.timestamp)}</div>
            </div>
          </div>
        ))}
        
        {isTyping && (
          <div className="message-container agent">
            <div className="message-content">
              <div className="message typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <div className="chat-input">
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Message Alex..."
          disabled={!isConnected}
        />
        <button onClick={sendMessage} disabled={!isConnected || !inputMessage.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}

export default App;
