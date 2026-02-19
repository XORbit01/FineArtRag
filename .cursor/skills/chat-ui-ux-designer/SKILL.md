---
name: chat-ui-ux-designer
description: Designs beautiful, modern UI/UX for real-time chat web applications with WebSocket integration. Creates interactive chat interfaces, message components, typing indicators, notifications, and responsive layouts. Use when building chat apps, messaging interfaces, real-time communication features, or WebSocket-based applications.
---

# Chat UI/UX Designer - WebSocket Chat Apps

## Quick Start

Designing effective chat interfaces requires:
1. **Real-time Communication**: WebSocket integration for instant messaging
2. **Message Display**: Clear, readable message bubbles and threads
3. **User Feedback**: Typing indicators, read receipts, delivery status
4. **Responsive Design**: Works on desktop, tablet, and mobile
5. **Accessibility**: Keyboard navigation, screen reader support, ARIA labels

## Architecture Overview

```
User Input → WebSocket Client → Server → WebSocket Broadcast → All Clients → UI Update
```

## Core Components

### 1. WebSocket Connection Setup

**React with Socket.io:**
```jsx
import { useEffect, useState } from 'react';
import io from 'socket.io-client';

function ChatApp() {
  const [socket, setSocket] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Connect to WebSocket server
    const newSocket = io('http://localhost:3001', {
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5
    });

    newSocket.on('connect', () => {
      setIsConnected(true);
      console.log('Connected to server');
    });

    newSocket.on('disconnect', () => {
      setIsConnected(false);
    });

    newSocket.on('message', (message) => {
      setMessages((prev) => [...prev, message]);
    });

    setSocket(newSocket);

    return () => newSocket.close();
  }, []);

  return (
    <div className="chat-container">
      <ConnectionStatus isConnected={isConnected} />
      <MessageList messages={messages} />
      <MessageInput socket={socket} />
    </div>
  );
}
```

**Vanilla JavaScript with WebSocket API:**
```javascript
class ChatWebSocket {
  constructor(url) {
    this.url = url;
    this.socket = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  connect() {
    this.socket = new WebSocket(this.url);

    this.socket.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.onConnect();
    };

    this.socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.onMessage(message);
    };

    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.onError(error);
    };

    this.socket.onclose = () => {
      console.log('WebSocket disconnected');
      this.onDisconnect();
      this.attemptReconnect();
    };
  }

  send(message) {
    if (this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => this.connect(), 1000 * this.reconnectAttempts);
    }
  }

  onConnect() {}
  onMessage(message) {}
  onError(error) {}
  onDisconnect() {}
}
```

### 2. Message Display Component

**Modern Message Bubble Design:**
```jsx
import React from 'react';
import './MessageBubble.css';

function MessageBubble({ message, isOwn, timestamp, status }) {
  return (
    <div className={`message-bubble ${isOwn ? 'own' : 'other'}`}>
      {!isOwn && (
        <div className="message-avatar">
          <img src={message.avatar} alt={message.username} />
        </div>
      )}
      
      <div className="message-content">
        {!isOwn && <div className="message-username">{message.username}</div>}
        
        <div className="message-text">{message.text}</div>
        
        <div className="message-footer">
          <span className="message-time">{formatTime(timestamp)}</span>
          {isOwn && (
            <span className={`message-status ${status}`}>
              {status === 'sent' && '✓'}
              {status === 'delivered' && '✓✓'}
              {status === 'read' && '✓✓'}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// CSS
.message-bubble {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
  animation: slideIn 0.3s ease-out;
}

.message-bubble.own {
  flex-direction: row-reverse;
}

.message-content {
  max-width: 70%;
  padding: 12px 16px;
  border-radius: 18px;
  background: var(--bg-own-message, #007bff);
  color: white;
  position: relative;
}

.message-bubble.other .message-content {
  background: var(--bg-other-message, #e9ecef);
  color: #212529;
}

.message-footer {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 4px;
  font-size: 0.75rem;
  opacity: 0.8;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
```

### 3. Typing Indicator

**Real-time Typing Feedback:**
```jsx
function TypingIndicator({ users }) {
  if (users.length === 0) return null;

  return (
    <div className="typing-indicator">
      <div className="typing-dots">
        <span></span>
        <span></span>
        <span></span>
      </div>
      <span className="typing-text">
        {users.length === 1 
          ? `${users[0]} is typing...`
          : `${users.length} people are typing...`}
      </span>
    </div>
  );
}

// WebSocket integration
useEffect(() => {
  const typingTimeout = setTimeout(() => {
    socket.emit('stop-typing', { userId });
  }, 1000);

  socket.on('user-typing', (data) => {
    setTypingUsers((prev) => {
      if (!prev.includes(data.username)) {
        return [...prev, data.username];
      }
      return prev;
    });
  });

  socket.on('user-stopped-typing', (data) => {
    setTypingUsers((prev) => prev.filter(u => u !== data.username));
  });

  return () => clearTimeout(typingTimeout);
}, [inputValue]);

// CSS
.typing-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  color: #6c757d;
  font-size: 0.875rem;
}

.typing-dots {
  display: flex;
  gap: 4px;
}

.typing-dots span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #6c757d;
  animation: typing 1.4s infinite;
}

.typing-dots span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-dots span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0%, 60%, 100% {
    transform: translateY(0);
    opacity: 0.7;
  }
  30% {
    transform: translateY(-10px);
    opacity: 1;
  }
}
```

### 4. Message Input Component

**Rich Input with Features:**
```jsx
function MessageInput({ socket, onSend }) {
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const textareaRef = useRef(null);

  const handleInput = (e) => {
    const value = e.target.value;
    setInput(value);

    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }

    // Typing indicator
    if (!isTyping && value.length > 0) {
      setIsTyping(true);
      socket.emit('typing', { userId });
    }

    // Debounce stop typing
    clearTimeout(typingTimeout);
    typingTimeout = setTimeout(() => {
      setIsTyping(false);
      socket.emit('stop-typing', { userId });
    }, 1000);
  };

  const handleSend = (e) => {
    e.preventDefault();
    if (input.trim()) {
      socket.emit('message', {
        text: input.trim(),
        timestamp: Date.now(),
        userId
      });
      setInput('');
      setIsTyping(false);
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend(e);
    }
  };

  return (
    <form className="message-input-container" onSubmit={handleSend}>
      <div className="input-wrapper">
        <textarea
          ref={textareaRef}
          className="message-input"
          value={input}
          onChange={handleInput}
          onKeyPress={handleKeyPress}
          placeholder="Type a message..."
          rows={1}
          maxLength={2000}
        />
        <button 
          type="submit" 
          className="send-button"
          disabled={!input.trim()}
        >
          <SendIcon />
        </button>
      </div>
      <div className="input-footer">
        <span className="char-count">{input.length}/2000</span>
        <span className="hint">Press Enter to send, Shift+Enter for new line</span>
      </div>
    </form>
  );
}

// CSS
.message-input-container {
  padding: 16px;
  background: white;
  border-top: 1px solid #e9ecef;
}

.input-wrapper {
  display: flex;
  gap: 8px;
  align-items: flex-end;
}

.message-input {
  flex: 1;
  padding: 12px 16px;
  border: 2px solid #e9ecef;
  border-radius: 24px;
  resize: none;
  font-family: inherit;
  font-size: 1rem;
  max-height: 120px;
  transition: border-color 0.2s;
}

.message-input:focus {
  outline: none;
  border-color: #007bff;
}

.send-button {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  background: #007bff;
  color: white;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
}

.send-button:hover:not(:disabled) {
  background: #0056b3;
}

.send-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

### 5. Connection Status Indicator

**Visual Connection Feedback:**
```jsx
function ConnectionStatus({ isConnected, reconnectAttempts }) {
  return (
    <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
      <div className="status-dot"></div>
      <span>
        {isConnected 
          ? 'Connected' 
          : `Reconnecting... (${reconnectAttempts}/5)`}
      </span>
    </div>
  );
}

// CSS
.connection-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  font-size: 0.875rem;
  background: #f8f9fa;
  border-bottom: 1px solid #e9ecef;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

.connection-status.connected .status-dot {
  background: #28a745;
}

.connection-status.disconnected .status-dot {
  background: #dc3545;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}
```

## UI/UX Best Practices

### Layout Design

**Chat Container Structure:**
```jsx
<div className="chat-app">
  {/* Header */}
  <header className="chat-header">
    <div className="header-content">
      <Avatar src={chatAvatar} />
      <div className="header-info">
        <h2>{chatName}</h2>
        <span className="online-status">{onlineStatus}</span>
      </div>
    </div>
    <div className="header-actions">
      <button aria-label="Search">🔍</button>
      <button aria-label="More options">⋯</button>
    </div>
  </header>

  {/* Messages Area */}
  <main className="messages-container">
    <div className="messages-list" ref={messagesEndRef}>
      {messages.map(msg => (
        <MessageBubble key={msg.id} {...msg} />
      ))}
    </div>
    <TypingIndicator users={typingUsers} />
  </main>

  {/* Input Area */}
  <footer className="chat-footer">
    <MessageInput socket={socket} />
  </footer>
</div>
```

### Responsive Design

**Mobile-First Approach:**
```css
.chat-app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  max-width: 100%;
}

/* Desktop */
@media (min-width: 768px) {
  .chat-app {
    max-width: 800px;
    margin: 0 auto;
    box-shadow: 0 0 20px rgba(0,0,0,0.1);
  }
}

/* Mobile */
@media (max-width: 767px) {
  .message-content {
    max-width: 85%;
  }
  
  .chat-header {
    padding: 12px;
  }
  
  .message-input-container {
    padding: 12px;
  }
}
```

### Accessibility

**ARIA Labels and Keyboard Navigation:**
```jsx
<div 
  className="messages-list"
  role="log"
  aria-label="Chat messages"
  aria-live="polite"
  aria-atomic="false"
>
  {messages.map(msg => (
    <div
      key={msg.id}
      role="article"
      aria-label={`Message from ${msg.username} at ${formatTime(msg.timestamp)}`}
    >
      <MessageBubble {...msg} />
    </div>
  ))}
</div>

<button
  className="send-button"
  type="submit"
  aria-label="Send message"
  aria-disabled={!input.trim()}
>
  <SendIcon aria-hidden="true" />
</button>
```

### Performance Optimization

**Virtual Scrolling for Long Message Lists:**
```jsx
import { useVirtualizer } from '@tanstack/react-virtual';

function VirtualizedMessageList({ messages }) {
  const parentRef = useRef();

  const virtualizer = useVirtualizer({
    count: messages.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 80,
    overscan: 5,
  });

  return (
    <div ref={parentRef} className="messages-container" style={{ height: '100%', overflow: 'auto' }}>
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {virtualizer.getVirtualItems().map((virtualItem) => (
          <div
            key={virtualItem.key}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: `${virtualItem.size}px`,
              transform: `translateY(${virtualItem.start}px)`,
            }}
          >
            <MessageBubble {...messages[virtualItem.index]} />
          </div>
        ))}
      </div>
    </div>
  );
}
```

## Common Patterns

### Message Reactions

```jsx
function MessageReactions({ messageId, reactions, onReact }) {
  return (
    <div className="message-reactions">
      {Object.entries(reactions).map(([emoji, users]) => (
        <button
          key={emoji}
          className="reaction-button"
          onClick={() => onReact(messageId, emoji)}
          aria-label={`React with ${emoji}`}
        >
          <span>{emoji}</span>
          <span className="reaction-count">{users.length}</span>
        </button>
      ))}
      <button className="add-reaction" aria-label="Add reaction">
        +
      </button>
    </div>
  );
}
```

### Read Receipts

```jsx
function ReadReceipt({ messageId, readBy }) {
  return (
    <div className="read-receipts">
      {readBy.map(user => (
        <img
          key={user.id}
          src={user.avatar}
          alt={user.name}
          className="read-avatar"
          title={`Read by ${user.name}`}
        />
      ))}
    </div>
  );
}
```

### File Upload Preview

```jsx
function FileUploadPreview({ files, onRemove }) {
  return (
    <div className="file-preview-container">
      {files.map((file, index) => (
        <div key={index} className="file-preview">
          {file.type.startsWith('image/') ? (
            <img src={URL.createObjectURL(file)} alt="Preview" />
          ) : (
            <div className="file-icon">{getFileIcon(file.type)}</div>
          )}
          <span className="file-name">{file.name}</span>
          <button onClick={() => onRemove(index)} aria-label="Remove file">
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
```

### Notification Badge

```jsx
function NotificationBadge({ count }) {
  if (count === 0) return null;
  
  return (
    <span className="notification-badge" aria-label={`${count} unread messages`}>
      {count > 99 ? '99+' : count}
    </span>
  );
}

// CSS
.notification-badge {
  position: absolute;
  top: -8px;
  right: -8px;
  background: #dc3545;
  color: white;
  border-radius: 12px;
  padding: 2px 8px;
  font-size: 0.75rem;
  font-weight: bold;
  min-width: 20px;
  text-align: center;
}
```

## Design System

### Color Palette

```css
:root {
  /* Primary Colors */
  --primary: #007bff;
  --primary-hover: #0056b3;
  --primary-light: #e7f3ff;
  
  /* Message Colors */
  --own-message-bg: #007bff;
  --own-message-text: #ffffff;
  --other-message-bg: #e9ecef;
  --other-message-text: #212529;
  
  /* Status Colors */
  --success: #28a745;
  --warning: #ffc107;
  --error: #dc3545;
  --info: #17a2b8;
  
  /* Neutral Colors */
  --bg-primary: #ffffff;
  --bg-secondary: #f8f9fa;
  --border-color: #e9ecef;
  --text-primary: #212529;
  --text-secondary: #6c757d;
  
  /* Spacing */
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;
  
  /* Border Radius */
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 18px;
  --radius-full: 9999px;
}
```

### Typography

```css
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
  font-size: 16px;
  line-height: 1.5;
  color: var(--text-primary);
}

.message-text {
  font-size: 1rem;
  line-height: 1.5;
  word-wrap: break-word;
}

.message-username {
  font-size: 0.875rem;
  font-weight: 600;
  margin-bottom: 4px;
}

.message-time {
  font-size: 0.75rem;
  opacity: 0.7;
}
```

## Required Dependencies

```bash
# React + Socket.io
npm install react react-dom socket.io-client

# Or Vue + Socket.io
npm install vue socket.io-client

# Or Vanilla JS
# Just use native WebSocket API

# Virtual Scrolling (optional)
npm install @tanstack/react-virtual

# Styling
npm install styled-components  # or use CSS modules
```

## Additional Resources

- For WebSocket server setup, see [server-setup.md](server-setup.md)
- For advanced animations, see [animations.md](animations.md)
- For accessibility guidelines, see [accessibility.md](accessibility.md)
