"use strict";

const state = {
  ws: null,
  activeChatId: null,
  connected: false,
  processing: false,
  chats: [],
  typingNode: null,
};

const ui = {
  newChatBtn: document.getElementById("newChatBtn"),
  resetBtn: document.getElementById("resetBtn"),
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
  chatTabs: document.getElementById("chatTabs"),
  stageHeader: document.getElementById("stageHeader"),
  chatLog: document.getElementById("chatLog"),
  composer: document.getElementById("composer"),
  messageInput: document.getElementById("messageInput"),
  sendBtn: document.getElementById("sendBtn"),
  msgTemplate: document.getElementById("msgTemplate"),
};

function setStatus(text, isLive = false) {
  ui.statusText.textContent = text;
  ui.statusDot.classList.toggle("live", isLive);
}

function apiBaseUrl() {
  const proto = window.location.protocol === "https:" ? "https" : "http";
  const host = window.location.hostname || "127.0.0.1";
  return `${proto}://${host}:8000`;
}

function wsUrl() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.hostname || "127.0.0.1";
  return `${proto}://${host}:8000/ws/chat`;
}

function addMessage(role, text, sources = []) {
  const node = ui.msgTemplate.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  node.querySelector(".role").textContent = role === "user" ? "You" : "Atelier";
  node.querySelector(".text").textContent = text;

  const sourcesList = node.querySelector(".sources");
  renderSources(sourcesList, sources);

  ui.chatLog.appendChild(node);
  ui.chatLog.scrollTop = ui.chatLog.scrollHeight;
  return node;
}

function ensureTypingIndicator() {
  if (state.typingNode) return;
  const node = addMessage("assistant", "");
  node.classList.add("typing");
  const textNode = node.querySelector(".text");
  textNode.innerHTML = `
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
  `;
  state.typingNode = node;
}

function clearTypingIndicator() {
  if (!state.typingNode) return;
  state.typingNode.remove();
  state.typingNode = null;
}

async function addTypedAssistantMessage(text, sources = []) {
  const node = addMessage("assistant", "", []);
  const textEl = node.querySelector(".text");
  const content = String(text || "");
  if (!content) {
    return;
  }

  let i = 0;
  const chunk = Math.max(1, Math.ceil(content.length / 220));
  while (i < content.length) {
    i = Math.min(content.length, i + chunk);
    textEl.textContent = content.slice(0, i);
    ui.chatLog.scrollTop = ui.chatLog.scrollHeight;
    // Small delay for "live typing" feel.
    await new Promise((r) => setTimeout(r, 14));
  }

  const sourcesList = node.querySelector(".sources");
  renderSources(sourcesList, sources);
  ui.chatLog.scrollTop = ui.chatLog.scrollHeight;
}

function renderSources(sourcesList, sources) {
  if (!sourcesList) return;
  sourcesList.innerHTML = "";
  if (!Array.isArray(sources) || !sources.length) {
    sourcesList.hidden = true;
    return;
  }
  sourcesList.hidden = false;
  sources.forEach((s) => {
    const li = document.createElement("li");
    const sourceUrl = s.url || s.source_url || "";
    const sourceFile = s.source_file || s.sourceFile || "source";
    const label = sourceUrl || sourceFile;
    if (sourceUrl) {
      let pretty = label;
      try {
        const u = new URL(sourceUrl);
        pretty = `${u.hostname}${u.pathname}`.replace(/\/$/, "");
      } catch (err) {
        // keep original label
      }
      const a = document.createElement("a");
      a.href = sourceUrl;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.className = "source-link";
      a.title = sourceUrl;
      a.textContent = `link: ${pretty}`;
      li.appendChild(a);
    } else {
      li.textContent = label;
    }
    sourcesList.appendChild(li);
  });
}

function updateStageTitle() {
  const chat = state.chats.find((c) => c.chat_id === state.activeChatId);
  const h2 = ui.stageHeader.querySelector("h2");
  if (chat && h2) {
    h2.textContent = chat.title || "Studio Conversation";
  }
}

function toChatTitle(message) {
  return String(message || "").trim().replace(/\s+/g, " ").slice(0, 56);
}

function applyFirstPromptTitle(message) {
  const chat = state.chats.find((c) => c.chat_id === state.activeChatId);
  if (!chat) return;
  const isUntitled = !chat.title || chat.title === "Untitled chat";
  const noTurns = Number(chat.turn_count || 0) === 0;
  if (!isUntitled || !noTurns) return;
  const nextTitle = toChatTitle(message);
  if (!nextTitle) return;
  chat.title = nextTitle;
  renderTabs();
  updateStageTitle();
}

function renderTabs() {
  ui.chatTabs.innerHTML = "";
  if (!state.chats.length) return;

  state.chats.forEach((chat) => {
    const tab = document.createElement("div");
    tab.className = `chat-tab${chat.chat_id === state.activeChatId ? " active" : ""}`;
    tab.dataset.chatId = chat.chat_id;
    tab.innerHTML = `
      <div class="chat-tab-row">
        <button class="chat-tab-main" type="button">
          <span class="title">${chat.title || "Untitled chat"}</span>
          <span class="meta">${chat.turn_count || 0} turns</span>
        </button>
        <button class="chat-delete" type="button" title="Delete chat" aria-label="Delete chat">
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M9 3h6l1 2h4v2H4V5h4l1-2Zm1 6h2v9h-2V9Zm4 0h2v9h-2V9ZM7 9h2v9H7V9Z"/>
          </svg>
        </button>
      </div>
    `;

    const mainBtn = tab.querySelector(".chat-tab-main");
    const delBtn = tab.querySelector(".chat-delete");

    mainBtn.addEventListener("click", async () => {
      await switchChat(chat.chat_id);
    });
    delBtn.addEventListener("click", async (event) => {
      event.stopPropagation();
      await deleteChat(chat.chat_id);
    });

    ui.chatTabs.appendChild(tab);
  });
}

async function deleteChat(chatId) {
  const chat = state.chats.find((c) => c.chat_id === chatId);
  const title = chat?.title || "this chat";
  const ok = window.confirm(`Delete "${title}"?`);
  if (!ok) return;

  await fetchJSON(`/v1/chats/${chatId}`, { method: "DELETE" });
  state.chats = state.chats.filter((c) => c.chat_id !== chatId);

  if (state.activeChatId === chatId) {
    if (state.chats.length) {
      state.activeChatId = state.chats[0].chat_id;
      renderTabs();
      await switchChat(state.activeChatId);
    } else {
      state.activeChatId = null;
      ui.chatLog.innerHTML = "";
      renderTabs();
      await createChat();
    }
  } else {
    renderTabs();
  }
}

async function fetchJSON(path, options = {}) {
  const res = await fetch(`${apiBaseUrl()}${path}`, options);
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || "Request failed");
  }
  return res.json();
}

async function loadChats() {
  state.chats = await fetchJSON("/v1/chats");
  renderTabs();
}

function formatTurnsToUI(detail) {
  ui.chatLog.innerHTML = "";
  clearTypingIndicator();
  (detail.turns || []).forEach((t) => {
    addMessage("user", t.question || "");
    addMessage("assistant", t.answer || "", t.sources || []);
  });
  updateStageTitle();
}

async function switchChat(chatId) {
  state.activeChatId = chatId;
  renderTabs();
  updateStageTitle();
  const detail = await fetchJSON(`/v1/chats/${chatId}`);
  formatTurnsToUI(detail);

  if (state.connected && state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "switch_chat", chat_id: chatId }));
  }
}

async function createChat() {
  const chat = await fetchJSON("/v1/chats", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  state.chats = [chat, ...state.chats];
  state.activeChatId = chat.chat_id;
  renderTabs();
  ui.chatLog.innerHTML = "";
  updateStageTitle();

  if (state.connected && state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "switch_chat", chat_id: chat.chat_id }));
  }
}

function setProcessing(isBusy) {
  state.processing = isBusy;
  ui.sendBtn.disabled = isBusy;
  ui.messageInput.disabled = isBusy;
  if (isBusy) {
    ensureTypingIndicator();
  } else {
    clearTypingIndicator();
  }
}

function connectWebSocket() {
  const socketUrl = wsUrl();

  if (state.ws) {
    try {
      state.ws.close();
    } catch (err) {
      console.warn(err);
    }
  }

  setStatus("Connecting...", false);
  const ws = new WebSocket(socketUrl);
  state.ws = ws;

  ws.addEventListener("open", () => {
    state.connected = true;
    setStatus("Connected", true);
    addMessage("assistant", "Live chat ready.");
  });

  ws.addEventListener("close", () => {
    state.connected = false;
    setStatus("Disconnected", false);
    setProcessing(false);
  });

  ws.addEventListener("error", () => {
    setStatus("WebSocket error", false);
  });

  ws.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);

    if (data.type === "ready") {
      if (!state.activeChatId) {
        state.activeChatId = data.chat_id || null;
      }
      loadChats().catch((err) => addMessage("assistant", `Error: ${err.message}`));
      if (state.activeChatId) {
        switchChat(state.activeChatId).catch((err) => addMessage("assistant", `Error: ${err.message}`));
      }
      return;
    }
    if (data.type === "new_chat_ok") {
      loadChats().catch((err) => addMessage("assistant", `Error: ${err.message}`));
      if (data.active_chat_id) {
        switchChat(data.active_chat_id).catch((err) => addMessage("assistant", `Error: ${err.message}`));
      }
      return;
    }
    if (data.type === "switch_chat_ok") {
      return;
    }
    if (data.type === "processing") {
      setProcessing(true);
      return;
    }
    if (data.type === "answer") {
      setProcessing(false);
      addTypedAssistantMessage(data.answer || "", data.sources || []);
      return;
    }
    if (data.type === "reset_ok") {
      addMessage("assistant", "Session memory cleared.");
      return;
    }
    if (data.type === "error") {
      setProcessing(false);
      addMessage("assistant", `Error: ${data.error || "Unknown error"}`);
    }
  });
}

async function httpFallbackQuery(chatId, message) {
  const res = await fetch(`${apiBaseUrl()}/v1/chats/${chatId}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: message,
      return_sources: true,
      use_memory: false,
    }),
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || "HTTP query failed");
  }
  return res.json();
}

async function sendMessage() {
  const message = ui.messageInput.value.trim();
  if (!message || state.processing) return;
  if (!state.activeChatId) {
    await createChat();
  }

  ui.messageInput.value = "";
  addMessage("user", message);
  applyFirstPromptTitle(message);

  if (state.connected && state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "switch_chat", chat_id: state.activeChatId }));
    state.ws.send(JSON.stringify({ type: "message", message }));
    return;
  }

  try {
    setProcessing(true);
    setStatus("HTTP fallback", false);
    const result = await httpFallbackQuery(state.activeChatId, message);
    setProcessing(false);
    await addTypedAssistantMessage(result.answer || "", result.sources || []);
  } catch (err) {
    setProcessing(false);
    addMessage("assistant", `Error: ${err.message}`);
  }
}

ui.newChatBtn.addEventListener("click", () => {
  createChat().catch((err) => addMessage("assistant", `Error: ${err.message}`));
});

ui.resetBtn.addEventListener("click", () => {
  if (!state.activeChatId) return;
  if (state.connected && state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "reset" }));
  } else {
    fetchJSON(`/v1/chats/${state.activeChatId}/reset`, { method: "POST" }).catch((err) =>
      addMessage("assistant", `Error: ${err.message}`)
    );
  }
  ui.chatLog.innerHTML = "";
  addMessage("assistant", "Current chat reset.");
  loadChats().catch((err) => addMessage("assistant", `Error: ${err.message}`));
});

ui.composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendMessage();
});

ui.messageInput.addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    await sendMessage();
  }
});

addMessage("assistant", "Welcome. Creating your studio chats...");
connectWebSocket();
loadChats()
  .then(async () => {
    if (!state.chats.length) {
      await createChat();
    } else {
      state.activeChatId = state.chats[0].chat_id;
      await switchChat(state.activeChatId);
    }
  })
  .catch((err) => addMessage("assistant", `Error: ${err.message}`));
