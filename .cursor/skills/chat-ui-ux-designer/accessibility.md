# Accessibility Guidelines for Chat Apps

## ARIA Labels and Roles

**Message List:**
```jsx
<div
  role="log"
  aria-label="Chat messages"
  aria-live="polite"
  aria-atomic="false"
  className="messages-list"
>
  {messages.map(msg => (
    <article
      key={msg.id}
      role="article"
      aria-label={`Message from ${msg.username}`}
    >
      <MessageBubble {...msg} />
    </article>
  ))}
</div>
```

**Input Field:**
```jsx
<textarea
  aria-label="Type your message"
  aria-describedby="input-hint"
  aria-required="true"
  aria-invalid={hasError}
  placeholder="Type a message..."
/>

<span id="input-hint" className="sr-only">
  Press Enter to send, Shift+Enter for new line
</span>
```

**Buttons:**
```jsx
<button
  type="submit"
  aria-label="Send message"
  aria-disabled={!input.trim()}
  className="send-button"
>
  <SendIcon aria-hidden="true" />
  <span className="sr-only">Send</span>
</button>
```

## Keyboard Navigation

**Tab Order:**
```jsx
function ChatApp() {
  return (
    <div className="chat-app">
      {/* Header - first in tab order */}
      <header className="chat-header">
        <button tabIndex={0}>Settings</button>
        <button tabIndex={0}>Search</button>
      </header>

      {/* Messages - skip in tab order */}
      <main className="messages-container" tabIndex={-1}>
        <MessageList messages={messages} />
      </main>

      {/* Input - last in tab order */}
      <footer className="chat-footer">
        <MessageInput tabIndex={0} />
      </footer>
    </div>
  );
}
```

**Keyboard Shortcuts:**
```jsx
useEffect(() => {
  const handleKeyDown = (e) => {
    // Focus input with '/' key
    if (e.key === '/' && e.target.tagName !== 'TEXTAREA') {
      e.preventDefault();
      inputRef.current?.focus();
    }

    // Send message with Ctrl+Enter
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault();
      handleSend();
    }

    // Escape to clear input
    if (e.key === 'Escape') {
      setInput('');
      inputRef.current?.blur();
    }
  };

  window.addEventListener('keydown', handleKeyDown);
  return () => window.removeEventListener('keydown', handleKeyDown);
}, []);
```

## Screen Reader Support

**Announce New Messages:**
```jsx
function MessageList({ messages }) {
  const [announcement, setAnnouncement] = useState('');

  useEffect(() => {
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      setAnnouncement(
        `New message from ${lastMessage.username}: ${lastMessage.text}`
      );
    }
  }, [messages]);

  return (
    <>
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      >
        {announcement}
      </div>
      <div className="messages-list">
        {messages.map(msg => (
          <MessageBubble key={msg.id} {...msg} />
        ))}
      </div>
    </>
  );
}
```

**Screen Reader Only Content:**
```css
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}
```

## Focus Management

**Focus on New Messages:**
```jsx
const messageRefs = useRef({});

useEffect(() => {
  if (messages.length > 0) {
    const lastMessageId = messages[messages.length - 1].id;
    const lastMessageElement = messageRefs.current[lastMessageId];
    
    if (lastMessageElement && document.activeElement === inputRef.current) {
      // Don't steal focus if user is typing
      return;
    }
    
    lastMessageElement?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}, [messages]);
```

**Skip to Content:**
```jsx
function SkipToContent() {
  return (
    <a
      href="#messages-list"
      className="skip-link"
      onClick={(e) => {
        e.preventDefault();
        document.getElementById('messages-list')?.focus();
      }}
    >
      Skip to messages
    </a>
  );
}

// CSS
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  background: #000;
  color: #fff;
  padding: 8px;
  text-decoration: none;
  z-index: 100;
}

.skip-link:focus {
  top: 0;
}
```

## Color Contrast

**WCAG AA Compliance:**
```css
/* Minimum contrast ratios */
:root {
  /* Text on primary background */
  --primary-text: #ffffff; /* 4.5:1 contrast on #007bff */
  
  /* Text on light background */
  --text-primary: #212529; /* 7:1 contrast on #ffffff */
  
  /* Text on gray background */
  --text-secondary: #495057; /* 4.5:1 contrast on #f8f9fa */
}

/* Error states */
.error-message {
  color: #721c24; /* High contrast for errors */
  background: #f8d7da;
  border: 1px solid #f5c6cb;
}
```

## Focus Indicators

**Visible Focus Styles:**
```css
*:focus {
  outline: 2px solid #007bff;
  outline-offset: 2px;
}

button:focus,
input:focus,
textarea:focus {
  outline: 2px solid #007bff;
  outline-offset: 2px;
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.25);
}

/* Remove default outline only if custom is provided */
*:focus:not(:focus-visible) {
  outline: none;
}
```

## Semantic HTML

**Proper Structure:**
```jsx
<main role="main">
  <header role="banner" className="chat-header">
    <h1>Chat Room</h1>
    <nav aria-label="Chat actions">
      <button aria-label="Search messages">Search</button>
      <button aria-label="Chat settings">Settings</button>
    </nav>
  </header>

  <section aria-label="Messages" className="messages-container">
    <div role="log" aria-live="polite" id="messages-list">
      {messages.map(msg => (
        <article key={msg.id}>
          <MessageBubble {...msg} />
        </article>
      ))}
    </div>
  </section>

  <footer role="contentinfo" className="chat-footer">
    <form aria-label="Send message">
      <MessageInput />
    </form>
  </footer>
</main>
```

## Error Handling

**Accessible Error Messages:**
```jsx
function MessageInput({ hasError, errorMessage }) {
  return (
    <div>
      <textarea
        aria-label="Type your message"
        aria-invalid={hasError}
        aria-describedby={hasError ? "error-message" : undefined}
      />
      {hasError && (
        <div
          id="error-message"
          role="alert"
          className="error-message"
        >
          {errorMessage}
        </div>
      )}
    </div>
  );
}
```

## Testing Accessibility

**Automated Testing:**
```bash
# Install axe-core
npm install --save-dev @axe-core/react

# Use in development
import React from 'react';
import { render } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

test('should not have accessibility violations', async () => {
  const { container } = render(<ChatApp />);
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});
```

**Manual Testing Checklist:**
- [ ] All interactive elements are keyboard accessible
- [ ] Focus indicators are visible
- [ ] Screen reader announces new messages
- [ ] Color contrast meets WCAG AA standards
- [ ] Forms have proper labels
- [ ] Error messages are announced
- [ ] Page structure uses semantic HTML
- [ ] Images have alt text
- [ ] ARIA labels are descriptive
