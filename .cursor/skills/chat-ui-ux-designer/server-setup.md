# WebSocket Server Setup for Chat Apps

## Node.js + Socket.io Server

**Basic Server:**
```javascript
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');

const app = express();
app.use(cors());

const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "http://localhost:3000",
    methods: ["GET", "POST"]
  }
});

// Store connected users
const users = new Map();

io.on('connection', (socket) => {
  console.log('User connected:', socket.id);

  // User joins chat
  socket.on('join', (userData) => {
    users.set(socket.id, userData);
    socket.broadcast.emit('user-joined', userData);
    
    // Send current users list
    io.emit('users-list', Array.from(users.values()));
  });

  // Handle messages
  socket.on('message', (messageData) => {
    const user = users.get(socket.id);
    const message = {
      ...messageData,
      id: Date.now().toString(),
      userId: socket.id,
      username: user?.username || 'Anonymous',
      timestamp: Date.now()
    };
    
    // Broadcast to all clients
    io.emit('message', message);
  });

  // Typing indicators
  socket.on('typing', (data) => {
    socket.broadcast.emit('user-typing', {
      userId: socket.id,
      username: users.get(socket.id)?.username
    });
  });

  socket.on('stop-typing', (data) => {
    socket.broadcast.emit('user-stopped-typing', {
      userId: socket.id,
      username: users.get(socket.id)?.username
    });
  });

  // Handle disconnection
  socket.on('disconnect', () => {
    const user = users.get(socket.id);
    users.delete(socket.id);
    socket.broadcast.emit('user-left', user);
    io.emit('users-list', Array.from(users.values()));
    console.log('User disconnected:', socket.id);
  });
});

const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

## Python + FastAPI + WebSockets

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
import json

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.users = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.users[websocket] = user_id

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        del self.users[websocket]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Broadcast message to all clients
            await manager.broadcast(json.dumps({
                "type": "message",
                "data": message_data,
                "sender": client_id
            }))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(json.dumps({
            "type": "user_left",
            "user_id": client_id
        }))
```

## Room-Based Chat

```javascript
// Join specific room
socket.on('join-room', (roomId) => {
  socket.join(roomId);
  socket.to(roomId).emit('user-joined-room', {
    userId: socket.id,
    roomId
  });
});

// Send message to room
socket.on('room-message', (data) => {
  io.to(data.roomId).emit('message', {
    ...data.message,
    roomId: data.roomId
  });
});

// Leave room
socket.on('leave-room', (roomId) => {
  socket.leave(roomId);
  socket.to(roomId).emit('user-left-room', {
    userId: socket.id,
    roomId
  });
});
```

## Message Persistence

```javascript
const mongoose = require('mongoose');

const MessageSchema = new mongoose.Schema({
  text: String,
  userId: String,
  username: String,
  roomId: String,
  timestamp: Date,
  readBy: [String]
});

const Message = mongoose.model('Message', MessageSchema);

// Save message
socket.on('message', async (messageData) => {
  const message = new Message({
    ...messageData,
    timestamp: new Date()
  });
  
  await message.save();
  
  // Broadcast to room
  io.to(messageData.roomId).emit('message', message);
});

// Load message history
socket.on('load-history', async (roomId) => {
  const messages = await Message.find({ roomId })
    .sort({ timestamp: -1 })
    .limit(50)
    .exec();
  
  socket.emit('message-history', messages.reverse());
});
```

## Rate Limiting

```javascript
const rateLimit = new Map();

function checkRateLimit(socketId) {
  const now = Date.now();
  const windowMs = 60000; // 1 minute
  const maxMessages = 30;
  
  if (!rateLimit.has(socketId)) {
    rateLimit.set(socketId, { count: 1, resetTime: now + windowMs });
    return true;
  }
  
  const limit = rateLimit.get(socketId);
  
  if (now > limit.resetTime) {
    limit.count = 1;
    limit.resetTime = now + windowMs;
    return true;
  }
  
  if (limit.count >= maxMessages) {
    return false;
  }
  
  limit.count++;
  return true;
}

socket.on('message', (messageData) => {
  if (!checkRateLimit(socket.id)) {
    socket.emit('error', { message: 'Rate limit exceeded' });
    return;
  }
  
  // Process message
});
```

## Authentication

```javascript
const jwt = require('jsonwebtoken');

io.use((socket, next) => {
  const token = socket.handshake.auth.token;
  
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    socket.userId = decoded.userId;
    socket.username = decoded.username;
    next();
  } catch (err) {
    next(new Error('Authentication failed'));
  }
});

io.on('connection', (socket) => {
  console.log('Authenticated user:', socket.username);
  // ... rest of connection logic
});
```
