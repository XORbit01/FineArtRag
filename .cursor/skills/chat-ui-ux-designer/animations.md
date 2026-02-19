# Chat UI Animations

## Message Entrance Animations

**Slide In from Bottom:**
```css
@keyframes slideInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message-bubble {
  animation: slideInUp 0.3s ease-out;
}
```

**Fade In with Scale:**
```css
@keyframes fadeInScale {
  from {
    opacity: 0;
    transform: scale(0.9);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.message-bubble {
  animation: fadeInScale 0.2s ease-out;
}
```

**Staggered Animation:**
```jsx
function MessageList({ messages }) {
  return (
    <div className="messages-list">
      {messages.map((msg, index) => (
        <div
          key={msg.id}
          className="message-bubble"
          style={{
            animationDelay: `${index * 0.05}s`
          }}
        >
          <MessageBubble {...msg} />
        </div>
      ))}
    </div>
  );
}
```

## Loading States

**Skeleton Loading:**
```jsx
function MessageSkeleton() {
  return (
    <div className="message-skeleton">
      <div className="skeleton-avatar"></div>
      <div className="skeleton-content">
        <div className="skeleton-line short"></div>
        <div className="skeleton-line"></div>
      </div>
    </div>
  );
}

// CSS
.skeleton-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
}

.skeleton-line {
  height: 12px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 6px;
  margin-bottom: 8px;
}

.skeleton-line.short {
  width: 60%;
}

@keyframes shimmer {
  0% {
    background-position: -200% 0;
  }
  100% {
    background-position: 200% 0;
  }
}
```

## Smooth Scrolling

**Auto-scroll to Bottom:**
```jsx
const messagesEndRef = useRef(null);

const scrollToBottom = () => {
  messagesEndRef.current?.scrollIntoView({ 
    behavior: 'smooth',
    block: 'end'
  });
};

useEffect(() => {
  scrollToBottom();
}, [messages]);

// CSS
.messages-container {
  scroll-behavior: smooth;
  scrollbar-width: thin;
  scrollbar-color: #cbd5e0 transparent;
}

.messages-container::-webkit-scrollbar {
  width: 8px;
}

.messages-container::-webkit-scrollbar-track {
  background: transparent;
}

.messages-container::-webkit-scrollbar-thumb {
  background: #cbd5e0;
  border-radius: 4px;
}

.messages-container::-webkit-scrollbar-thumb:hover {
  background: #a0aec0;
}
```

## Button Interactions

**Send Button Pulse:**
```css
.send-button {
  transition: transform 0.2s, box-shadow 0.2s;
}

.send-button:active {
  transform: scale(0.95);
}

.send-button:not(:disabled):hover {
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(0, 123, 255, 0.4);
}
```

**Ripple Effect:**
```jsx
function RippleButton({ children, onClick }) {
  const [ripples, setRipples] = useState([]);

  const handleClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const newRipple = {
      id: Date.now(),
      x,
      y
    };
    
    setRipples([...ripples, newRipple]);
    
    setTimeout(() => {
      setRipples(ripples.filter(r => r.id !== newRipple.id));
    }, 600);
    
    onClick(e);
  };

  return (
    <button className="ripple-button" onClick={handleClick}>
      {children}
      {ripples.map(ripple => (
        <span
          key={ripple.id}
          className="ripple"
          style={{
            left: `${ripple.x}px`,
            top: `${ripple.y}px`
          }}
        />
      ))}
    </button>
  );
}

// CSS
.ripple-button {
  position: relative;
  overflow: hidden;
}

.ripple {
  position: absolute;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.6);
  transform: scale(0);
  animation: ripple-animation 0.6s ease-out;
  width: 20px;
  height: 20px;
  margin-left: -10px;
  margin-top: -10px;
}

@keyframes ripple-animation {
  to {
    transform: scale(4);
    opacity: 0;
  }
}
```

## Notification Animations

**Slide In Notification:**
```jsx
function Notification({ message, onClose }) {
  return (
    <div className="notification slide-in">
      <span>{message}</span>
      <button onClick={onClose}>×</button>
    </div>
  );
}

// CSS
.notification {
  position: fixed;
  top: 20px;
  right: 20px;
  padding: 16px 20px;
  background: #007bff;
  color: white;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  animation: slideInRight 0.3s ease-out;
}

@keyframes slideInRight {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}
```

## Transition Effects

**Page Transitions:**
```css
.chat-app {
  animation: fadeIn 0.3s ease-in;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
```

**Input Focus Animation:**
```css
.message-input {
  transition: border-color 0.2s, box-shadow 0.2s;
}

.message-input:focus {
  border-color: #007bff;
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
}
```

## Performance Tips

**Use CSS Transforms:**
```css
/* Good - GPU accelerated */
.message-bubble {
  transform: translateY(0);
  transition: transform 0.3s;
}

/* Avoid - causes reflow */
.message-bubble {
  top: 0;
  transition: top 0.3s;
}
```

**Reduce Motion for Accessibility:**
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```
