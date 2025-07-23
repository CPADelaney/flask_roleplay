// chat_page.js - Refactored for robust connection and conversation management
console.log("Chat page loaded!");

// ===== Utility Functions =====
function sanitizeAndRenderMarkdown(markdownText) {
  const renderedHTML = marked.parse(markdownText);
  return DOMPurify.sanitize(renderedHTML);
}

// ===== State Management =====
const AppState = {
  // User info
  userId: null,
  
  // Conversation state
  currentConvId: null,
  currentRoomId: null,  // Track which room we're actually in
  messagesOffset: 0,
  
  // UI state
  isDarkMode: false,
  isCreatingGame: false,
  isSelectingConversation: false,
  isSendingMessage: false,
  
  // Connection state
  socket: null,
  isConnected: false,
  reconnectionInProgress: false,
  
  // Message streaming
  currentAssistantBubble: null,
  partialAssistantText: "",
  
  // Constants
  MESSAGES_PER_LOAD: 20,
  NYX_SPACE_CONV_ID: "__nyx_space__"
};

// Universal updates object
let pendingUniversalUpdates = {
  roleplay_updates: {},
  npc_creations: [],
  npc_updates: [],
  character_stat_updates: { player_name: "Chase", stats: {} },
  relationship_updates: [],
  npc_introductions: [],
  location_creations: [],
  event_list_updates: [],
  inventory_updates: { player_name: "Chase", added_items: [], removed_items: [] },
  quest_updates: [],
  social_links: [],
  perk_unlocks: []
};

function resetPendingUniversalUpdates() {
  pendingUniversalUpdates = {
    roleplay_updates: {},
    npc_creations: [],
    npc_updates: [],
    character_stat_updates: { player_name: "Chase", stats: {} },
    relationship_updates: [],
    npc_introductions: [],
    location_creations: [],
    event_list_updates: [],
    inventory_updates: { player_name: "Chase", added_items: [], removed_items: [] },
    quest_updates: [],
    social_links: [],
    perk_unlocks: []
  };
}

// DOM Element cache
const DOM = {};

// ===== Socket Management =====
class SocketManager {
  constructor() {
    this.socket = null;
    this.handlers = {};
  }

  initialize() {
    if (this.socket) {
      console.warn("Socket already initialized");
      return;
    }

    // Create socket with robust configuration
    this.socket = createRobustSocketConnection({
      onConnect: (socket, wasReconnect) => this.handleConnect(socket, wasReconnect),
      onDisconnect: (socket, reason) => this.handleDisconnect(socket, reason),
      onReconnect: (socket, attemptNumber) => this.handleReconnect(socket, attemptNumber),
      onReconnectFailed: () => this.handleReconnectFailed()
    });

    AppState.socket = this.socket;
    this.setupMessageHandlers();
    this.setupHeartbeat();
  }

  handleConnect(socket, wasReconnect) {
    console.log("Socket connected with ID:", socket.id);
    AppState.isConnected = true;
    
    if (wasReconnect) {
      appendMessage({ sender: "system", content: "Connection restored!" }, true);
      AppState.reconnectionInProgress = false;
    }
    
    // Only join room if we have a conversation selected and we're not already in it
    if (AppState.currentConvId && AppState.currentConvId !== AppState.currentRoomId) {
      this.joinRoom(AppState.currentConvId);
    }
  }

  handleDisconnect(socket, reason) {
    console.error("Socket disconnected:", reason);
    AppState.isConnected = false;
    AppState.currentRoomId = null;  // Clear room state
    AppState.reconnectionInProgress = true;
    
    appendMessage({ 
      sender: "system", 
      content: `Connection lost (${reason}). Attempting to reconnect...` 
    }, true);
  }

  handleReconnect(socket, attemptNumber) {
    console.log(`Socket reconnected after ${attemptNumber} attempts`);
    AppState.isConnected = true;
    AppState.reconnectionInProgress = false;
    
    // Rejoin the conversation room if we had one
    if (AppState.currentConvId) {
      this.joinRoom(AppState.currentConvId);
    }
  }

  handleReconnectFailed() {
    console.error("Socket reconnection failed");
    appendMessage({ 
      sender: "system", 
      content: "Unable to reconnect. Please refresh the page." 
    }, true);
  }

  joinRoom(conversationId) {
    if (!this.socket || !this.socket.connected) {
      console.warn("Cannot join room - socket not connected");
      return false;
    }

    if (AppState.currentRoomId === conversationId) {
      console.log(`Already in room ${conversationId}`);
      return true;
    }

    console.log(`Joining room ${conversationId}`);
    this.socket.emit('join', { conversation_id: conversationId });
    return true;
  }

  setupMessageHandlers() {
    // Room events
    this.socket.on("joined", (data) => {
      console.log("Joined room:", data.room);
      AppState.currentRoomId = data.room;
      
      // Only show connection message on initial join
      if (!AppState.isSendingMessage) {
        appendMessage({ sender: "system", content: "Connected to game session" }, true);
      }
    });

    // Message streaming events
    this.socket.on("new_token", (payload) => {
      handleNewToken(payload.token);
    });

    this.socket.on("done", (payload) => {
      console.log("Done streaming");
      finalizeAssistantMessage(payload.full_text);
      AppState.isSendingMessage = false;
    });

    this.socket.on("error", (payload) => {
      console.error("Server error:", payload.error);
      appendMessage({ 
        sender: "system", 
        content: `Error: ${payload.error}` 
      }, true);
      AppState.isSendingMessage = false;
    });

    this.socket.on("image", (payload) => {
      console.log("Received image:", payload);
      appendImageToChat(payload.image_url, payload.reason);
    });

    this.socket.on("processing", (data) => {
      console.log("Server is processing:", data.message);
      showProcessingIndicator();
    });

    this.socket.on("game_state_update", (data) => {
      console.log("Game state updated:", data);
    });

    this.socket.on("server_heartbeat", (data) => {
      console.log("Received server heartbeat:", data.timestamp);
    });
  }

  setupHeartbeat() {
    setInterval(() => {
      if (this.socket && this.socket.connected) {
        this.socket.emit('client_heartbeat', { timestamp: Date.now() });
      }
    }, 20000); // Every 20 seconds
  }

  sendMessage(data) {
    if (!this.socket || !this.socket.connected) {
      console.error("Cannot send message - socket not connected");
      return false;
    }

    if (AppState.currentRoomId !== String(data.conversation_id)) {
      console.warn("Not in the correct room, joining first");
      if (!this.joinRoom(data.conversation_id)) {
        return false;
      }
    }

    AppState.isSendingMessage = true;
    this.socket.emit("storybeat", data);
    return true;
  }
}

// Create global socket manager instance
const socketManager = new SocketManager();

// ===== Message Display Functions =====
function appendMessage(message, autoScroll = true) {
  const chatWindow = DOM.chatWindow || document.getElementById("chatWindow");
  const bubbleRow = createBubble(message);
  chatWindow.appendChild(bubbleRow);
  
  if (autoScroll) {
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }
  
  return bubbleRow;
}

function createBubble(message) {
  const row = document.createElement("div");
  row.classList.add("message-row");
  
  if (message.sender === "user") {
    row.classList.add("user-row");
  } else {
    row.classList.add("gpt-row");
  }
  
  const bubble = document.createElement("div");
  bubble.classList.add("message-bubble");
  
  const safeContent = sanitizeAndRenderMarkdown(message.content || "");
  bubble.innerHTML = `<strong>${DOMPurify.sanitize(message.sender)}:</strong> ${safeContent}`;
  
  row.appendChild(bubble);
  return row;
}

function appendImageToChat(imageUrl, reason) {
  const chatWindow = DOM.chatWindow || document.getElementById("chatWindow");
  const imageRow = document.createElement("div");
  imageRow.classList.add("message-row", "gpt-row");

  const imageBubble = document.createElement("div");
  imageBubble.classList.add("message-bubble", "image-bubble");
  imageBubble.innerHTML = `
    <div class="image-container">
      <img src="${imageUrl}" alt="Generated scene" style="max-width: 100%; border-radius: 5px;" />
      <div class="image-caption">
        ${DOMPurify.sanitize(reason || "AI-generated scene visualization")}
      </div>
    </div>
  `;
  
  imageRow.appendChild(imageBubble);
  chatWindow.appendChild(imageRow);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function handleNewToken(token) {
  removeProcessingIndicator();
  const chatWindow = DOM.chatWindow || document.getElementById("chatWindow");

  if (!AppState.currentAssistantBubble) {
    const row = document.createElement("div");
    row.classList.add("message-row", "gpt-row");
    
    const bubble = document.createElement("div");
    bubble.classList.add("message-bubble");
    bubble.innerHTML = `<strong>Nyx:</strong> `;
    
    const contentSpan = document.createElement('span');
    contentSpan.innerHTML = sanitizeAndRenderMarkdown(token);
    bubble.appendChild(contentSpan);

    row.appendChild(bubble);
    chatWindow.appendChild(row);
    
    AppState.currentAssistantBubble = contentSpan;
    AppState.partialAssistantText = token;
  } else {
    AppState.partialAssistantText += token;
    AppState.currentAssistantBubble.innerHTML = sanitizeAndRenderMarkdown(AppState.partialAssistantText);
  }
  
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function finalizeAssistantMessage(finalText) {
  if (!AppState.currentAssistantBubble) {
    if (finalText && finalText.trim() !== "") {
      console.warn("No current assistant bubble, creating new one for final text");
      appendMessage({ sender: "Nyx", content: finalText }, true);
    }
  } else {
    AppState.currentAssistantBubble.innerHTML = sanitizeAndRenderMarkdown(finalText);
  }
  
  AppState.currentAssistantBubble = null;
  AppState.partialAssistantText = "";
  
  const chatWindow = DOM.chatWindow || document.getElementById("chatWindow");
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function showProcessingIndicator() {
  const chatWindow = DOM.chatWindow || document.getElementById("chatWindow");
  removeProcessingIndicator();
  
  const processingDiv = document.createElement("div");
  processingDiv.id = "processingIndicator";
  processingDiv.innerHTML = '<div style="text-align: center; padding: 10px; font-style: italic; color: #888;">Processing your request...</div>';
  chatWindow.appendChild(processingDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function removeProcessingIndicator() {
  const indicator = document.getElementById("processingIndicator");
  if (indicator) {
    indicator.remove();
  }
}

// ===== User Actions =====
async function sendMessage() {
  const userInput = DOM.userMsgInput || document.getElementById("userMsg");
  const userText = userInput.value.trim();
  
  if (!userText || !AppState.currentConvId) {
    return;
  }

  // Prevent double sending
  if (AppState.isSendingMessage) {
    console.warn("Already sending a message");
    return;
  }

  userInput.value = "";

  // Handle Nyx Space differently
  if (AppState.currentConvId === AppState.NYX_SPACE_CONV_ID) {
    await handleNyxSpaceMessage(userText);
    return;
  }

  // Display user message
  appendMessage({ sender: "user", content: userText }, true);

  // Reset streaming state
  AppState.currentAssistantBubble = null;
  AppState.partialAssistantText = "";

  // Prepare message data
  const messageData = {
    user_input: userText,
    conversation_id: AppState.currentConvId,
    player_name: "Chase",
    advance_time: false,
    universal_update: pendingUniversalUpdates
  };

  // Send message
  console.log(`Sending message for conversation ${AppState.currentConvId}`);
  
  if (!socketManager.sendMessage(messageData)) {
    appendMessage({ 
      sender: "system", 
      content: "Failed to send message. Please check your connection." 
    }, true);
    AppState.isSendingMessage = false;
    return;
  }

  resetPendingUniversalUpdates();

  // Set timeout for response
  setTimeout(() => {
    if (AppState.isSendingMessage) {
      appendMessage({ 
        sender: "system", 
        content: "Server is taking longer than expected. Please wait..." 
      }, true);
    }
  }, 15000);
}

async function handleNyxSpaceMessage(userText) {
  try {
    // Save user message
    await fetch('/nyx_space/messages', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        sender: "user", 
        content: userText, 
        timestamp: Date.now()
      })
    });

    appendMessage({sender: "user", content: userText}, true);

    // Get Nyx's response
    const adminRequest = {
      user_input: userText,
      generate_response: true
    };
    
    const replyRes = await fetch('/admin/nyx_direct', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(adminRequest)
    });

    const replyData = await replyRes.json();
    console.log("Admin Nyx response:", replyData);

    // Extract response
    let aiReply = "...";
    if (replyData.response_result && replyData.response_result.main_message) {
      aiReply = replyData.response_result.main_message;
    } else if (replyData.processing_result && replyData.processing_result.message) {
      aiReply = replyData.processing_result.message;
    } else if (replyData.response_result && replyData.response_result.message) {
      aiReply = replyData.response_result.message;
    }

    appendMessage({sender: "Nyx", content: aiReply}, true);

    // Save Nyx's reply
    await fetch('/nyx_space/messages', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        sender: "Nyx", 
        content: aiReply, 
        timestamp: Date.now()
      })
    });
  } catch (error) {
    console.error("Error processing Nyx message:", error);
    appendMessage({
      sender: "Nyx", 
      content: "Sorry, an error occurred processing your message."
    }, true);
  }
}

function advanceTime() {
  if (!AppState.currentConvId) {
    alert("Please select a conversation first");
    return;
  }

  appendMessage({ 
    sender: "user", 
    content: "Let's advance to the next time period." 
  }, true);

  // Reset streaming state
  AppState.currentAssistantBubble = null;
  AppState.partialAssistantText = "";

  const messageData = {
    user_input: "Let's advance to the next time period.",
    conversation_id: AppState.currentConvId,
    player_name: "Chase",
    advance_time: true,
    universal_update: pendingUniversalUpdates
  };

  if (!socketManager.sendMessage(messageData)) {
    appendMessage({ 
      sender: "system", 
      content: "Failed to advance time. Please check your connection." 
    }, true);
    return;
  }

  resetPendingUniversalUpdates();
}

// ===== Conversation Management =====
async function selectConversation(convId) {
  if (AppState.isSelectingConversation) {
    console.log("Already selecting a conversation");
    return;
  }

  AppState.isSelectingConversation = true;
  AppState.currentConvId = convId;
  AppState.messagesOffset = 0;

  console.log(`Selecting conversation: ${convId}`);

  // Join the room
  if (socketManager.socket && socketManager.socket.connected) {
    socketManager.joinRoom(convId);
  }

  // Handle different conversation types
  if (convId === AppState.NYX_SPACE_CONV_ID) {
    await loadNyxSpace();
  } else {
    await loadGameConversation(convId);
  }

  AppState.isSelectingConversation = false;
}

async function loadNyxSpace() {
  const chatWindow = DOM.chatWindow || document.getElementById("chatWindow");
  chatWindow.innerHTML = "";
  
  appendMessage({sender: "system", content: "Loading Nyx's Space..."}, true);
  
  try {
    const res = await fetch("/nyx_space/messages");
    if (res.ok) {
      const data = await res.json();
      chatWindow.innerHTML = "";
      
      if (data.messages && data.messages.length > 0) {
        data.messages.forEach(msg => appendMessage(msg, false));
        chatWindow.scrollTop = chatWindow.scrollHeight;
      } else {
        appendMessage({
          sender: "Nyx", 
          content: "Welcome to Nyx's Space! You can chat with me here anytime."
        }, true);
      }
    } else {
      appendMessage({
        sender: "Nyx", 
        content: "Could not fetch Nyx's Space messages."
      }, true);
    }
  } catch (err) {
    console.error("Error loading Nyx Space:", err);
    chatWindow.innerHTML = "";
    appendMessage({
      sender: "Nyx", 
      content: "There was an error loading Nyx's Space."
    }, true);
  }
  
  // Hide game-specific buttons
  if (DOM.advanceTimeBtn) DOM.advanceTimeBtn.style.display = "none";
  if (DOM.loadMoreBtn) DOM.loadMoreBtn.style.display = "none";
}

async function loadGameConversation(convId) {
  // Show game-specific buttons
  if (DOM.advanceTimeBtn) DOM.advanceTimeBtn.style.display = "";
  if (DOM.loadMoreBtn) DOM.loadMoreBtn.style.display = "";
  
  await loadMessages(convId, true);
  await checkForWelcomeImage(convId);
}

async function loadMessages(convId, replace = false) {
  const chatWindow = DOM.chatWindow || document.getElementById("chatWindow");
  const loadMoreBtn = DOM.loadMoreBtn || document.getElementById("loadMore");
  
  const url = `/multiuser/conversations/${convId}/messages?offset=${AppState.messagesOffset}&limit=${AppState.MESSAGES_PER_LOAD}`;
  
  try {
    const res = await fetch(url, { method: "GET", credentials: "include" });
    if (!res.ok) {
      console.error("Failed to load messages:", res.status);
      appendMessage({
        sender: "system", 
        content: `Error loading messages for game ${convId}.`
      }, true);
      return;
    }
    
    const data = await res.json();
    
    if (replace) {
      // Clear chat window except for load more button
      while (chatWindow.firstChild && chatWindow.firstChild !== loadMoreBtn) {
        chatWindow.removeChild(chatWindow.firstChild);
      }
      if (!chatWindow.contains(loadMoreBtn)) {
        chatWindow.insertBefore(loadMoreBtn, chatWindow.firstChild);
      }
    }
    
    // Create messages
    const fragment = document.createDocumentFragment();
    data.messages.slice().reverse().forEach(msg => {
      fragment.appendChild(createBubble(msg));
    });

    if (replace) {
      chatWindow.appendChild(fragment);
      chatWindow.scrollTop = chatWindow.scrollHeight;
    } else {
      loadMoreBtn.after(fragment);
    }

    loadMoreBtn.style.display = data.messages.length < AppState.MESSAGES_PER_LOAD ? "none" : "block";
  } catch (err) {
    console.error("Error loading messages:", err);
    appendMessage({sender: "system", content: "Error loading messages."}, true);
  }
}

async function checkForWelcomeImage(convId) {
  try {
    const res = await fetch(`/universal/get_roleplay_value?conversation_id=${convId}&key=WelcomeImageUrl`, {
      method: "GET",
      credentials: "include"
    });
    if (res.ok) {
      const data = await res.json();
      if (data.value) {
        appendImageToChat(data.value, "Welcome to this new world");
      }
    }
  } catch (err) {
    console.error("Error checking for welcome image:", err);
  }
}

async function startNewGame() {
  if (AppState.isCreatingGame) {
    console.log("Game creation already in progress");
    return;
  }

  const newGameBtn = DOM.newGameBtn || document.getElementById("newGameBtn");
  if (newGameBtn) {
    newGameBtn.disabled = true;
    newGameBtn.textContent = "Creating...";
  }

  AppState.isCreatingGame = true;

  const chatWindow = DOM.chatWindow || document.getElementById("chatWindow");
  const loadingDiv = document.createElement("div");
  loadingDiv.id = "newGameLoadingIndicator";
  loadingDiv.innerHTML = '<div style="text-align: center; padding: 20px; font-style: italic; color: #888;">Initializing new game world...</div>';
  chatWindow.appendChild(loadingDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  try {
    const res = await fetch("/start_new_game", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({})
    });
    
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Failed to start new game");
    }

    AppState.currentConvId = data.conversation_id;
    AppState.messagesOffset = 0;

    // Update loading message
    loadingDiv.innerHTML = '<div style="text-align: center; padding: 20px; font-style: italic; color: #888;">Creating your world... This may take a minute...</div>';

    // Poll for completion
    const pollResult = await pollForGameReady(data.conversation_id);
    
    if (pollResult.ready) {
      loadingDiv.remove();
      
      // Join the new game room
      socketManager.joinRoom(AppState.currentConvId);
      
      // Load game content
      await loadMessages(AppState.currentConvId, true);
      await checkForWelcomeImage(AppState.currentConvId);
      await loadConversations();
      
      // Show opening narrative
      if (pollResult.opening_narrative) {
        appendMessage({ 
          sender: "Nyx", 
          content: pollResult.opening_narrative
        }, true);
      } else {
        appendMessage({ 
          sender: "system", 
          content: `New game started! Welcome to ${pollResult.conversation_name || "your new world"}.` 
        }, true);
      }
    } else {
      throw new Error(pollResult.error || "Game creation timed out");
    }

  } catch (err) {
    console.error("startNewGame error:", err);
    
    const existingLoadingDiv = document.getElementById("newGameLoadingIndicator");
    if (existingLoadingDiv) existingLoadingDiv.remove();
    
    appendMessage({ 
      sender: "system", 
      content: `Error starting new game: ${err.message}. Please try again.` 
    }, true);
  } finally {
    AppState.isCreatingGame = false;
    if (newGameBtn) {
      newGameBtn.disabled = false;
      newGameBtn.textContent = "New Game";
    }
  }
}

async function pollForGameReady(conversationId) {
  const maxAttempts = 60;
  let attempts = 0;

  while (attempts < maxAttempts) {
    attempts++;
    
    try {
      const statusRes = await fetch(`/new_game/conversation_status?conversation_id=${conversationId}`, {
        method: "GET",
        credentials: "include"
      });
      
      if (statusRes.ok) {
        const statusData = await statusRes.json();
        
        if (statusData.status === "ready") {
          return {
            ready: true,
            opening_narrative: statusData.opening_narrative,
            conversation_name: statusData.conversation_name
          };
        } else if (statusData.status === "failed") {
          return {
            ready: false,
            error: "Game creation failed"
          };
        }
      }
    } catch (pollError) {
      console.error("Error polling game status:", pollError);
    }
    
    // Wait before next attempt
    await new Promise(resolve => setTimeout(resolve, 3000));
  }
  
  return {
    ready: false,
    error: "Game creation timed out"
  };
}

async function loadConversations() {
  try {
    const res = await fetch("/multiuser/conversations", { 
      method: "GET", 
      credentials: "include" 
    });
    
    if (!res.ok) {
      console.error("Failed to get conversations:", res.status);
      return;
    }
    
    const convoData = await res.json();
    renderConvoList(convoData);
  } catch (err) {
    console.error("Error loading conversations:", err);
  }
}

function renderConvoList(conversations) {
  const convListDiv = DOM.convListDiv || document.getElementById("convList");
  convListDiv.innerHTML = "";
  
  conversations.forEach(conv => {
    const wrapper = document.createElement("div");
    wrapper.style.display = "flex";
    wrapper.style.marginBottom = "5px";
    
    const btn = document.createElement("button");
    btn.textContent = conv.name || `Game ${conv.id}`;
    btn.style.flex = "1";
    btn.dataset.convId = conv.id;

    // Click handler
    btn.addEventListener("click", async () => {
      if (AppState.isSelectingConversation) {
        return;
      }
      
      btn.disabled = true;
      try {
        await selectConversation(conv.id);
      } finally {
        btn.disabled = false;
      }
    });
    
    // Right-click handler
    btn.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      showContextMenu(e.clientX, e.clientY, conv.id);
    });
    
    wrapper.appendChild(btn);
    convListDiv.appendChild(wrapper);
  });
}

// ===== Context Menu =====
function showContextMenu(x, y, convId) {
  const menuDiv = DOM.contextMenuDiv || document.getElementById("contextMenu");
  menuDiv.innerHTML = "";

  const options = [
    { text: "Rename", action: () => renameConversation(convId) },
    { text: "Move to Folder", action: () => moveConversationToFolder(convId) },
    { text: "Delete", action: () => deleteConversation(convId) }
  ];

  options.forEach(opt => {
    const div = document.createElement("div");
    div.textContent = opt.text;
    div.addEventListener("click", (e) => {
      e.stopPropagation();
      opt.action();
      menuDiv.style.display = "none";
    });
    menuDiv.appendChild(div);
  });

  menuDiv.style.left = x + "px";
  menuDiv.style.top = y + "px";
  menuDiv.style.display = "block";
}

async function renameConversation(convId) {
  const newName = prompt("Enter new conversation name:");
  if (!newName || newName.trim() === "") return;
  
  try {
    const res = await fetch(`/multiuser/conversations/${convId}`, {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      credentials: "include",
      body: JSON.stringify({ conversation_name: newName.trim() })
    });
    
    if (res.ok) {
      await loadConversations();
    } else {
      const errData = await res.json().catch(() => ({error: "Unknown error"}));
      alert(`Error renaming conversation: ${errData.error}`);
    }
  } catch (err) {
    console.error("Rename conversation error:", err);
    alert("Network error renaming conversation.");
  }
}

async function moveConversationToFolder(convId) {
  const folderName = prompt("Enter folder name:");
  if (!folderName || folderName.trim() === "") return;
  
  try {
    const res = await fetch(`/multiuser/conversations/${convId}/move_folder`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder_name: folderName.trim() })
    });
    
    if (res.ok) {
      await loadConversations();
    } else {
      const errData = await res.json().catch(() => ({error: "Unknown error"}));
      alert(`Error moving conversation: ${errData.error}`);
    }
  } catch (err) {
    console.error("Move conversation error:", err);
    alert("Network error moving conversation.");
  }
}

async function deleteConversation(convId) {
  if (!confirm("Are you sure you want to delete this conversation? This cannot be undone.")) {
    return;
  }
  
  try {
    const res = await fetch(`/multiuser/conversations/${convId}`, {
      method: "DELETE",
      credentials: "include"
    });
    
    if (res.ok) {
      await loadConversations();
      if (AppState.currentConvId === convId) {
        const chatWindow = DOM.chatWindow || document.getElementById("chatWindow");
        chatWindow.innerHTML = '<span id="loadMore" style="display:none;">Load older messages...</span>';
        AppState.currentConvId = null;
        AppState.currentRoomId = null;
      }
    } else {
      const errData = await res.json().catch(() => ({error: "Unknown error"}));
      alert(`Error deleting conversation: ${errData.error}`);
    }
  } catch (err) {
    console.error("Delete conversation error:", err);
    alert("Network error deleting conversation.");
  }
}

// ===== Utility Functions =====
async function checkLoggedIn() {
  const res = await fetch("/whoami", { credentials: "include" });
  const data = await res.json();
  
  if (!data.logged_in) {
    window.location.href = "/login_page";
    return false;
  }
  
  AppState.userId = data.user_id;
  const logoutBtn = document.getElementById("logoutBtn");
  if (logoutBtn) {
    logoutBtn.style.display = "inline-block";
  }
  
  return true;
}

function loadDarkModeFromStorage() {
  const val = localStorage.getItem("dark_mode_enabled");
  if (val === "true") {
    AppState.isDarkMode = true;
    document.body.classList.add("dark-mode");
  } else {
    AppState.isDarkMode = false;
    document.body.classList.remove("dark-mode");
  }
}

function toggleDarkMode() {
  AppState.isDarkMode = !AppState.isDarkMode;
  localStorage.setItem("dark_mode_enabled", AppState.isDarkMode);
  document.body.classList.toggle("dark-mode", AppState.isDarkMode);
}

async function logout() {
  try {
    const res = await fetch("/logout", { method: "POST", credentials: "include" });
    if (res.ok) {
      window.location.href = "/login_page";
    } else {
      alert("Logout failed!");
    }
  } catch (err) {
    console.error("Logout error:", err);
  }
}

function loadPreviousMessages() {
  if (!AppState.currentConvId) return;
  AppState.messagesOffset += AppState.MESSAGES_PER_LOAD;
  loadMessages(AppState.currentConvId, false);
}

function attachEnterKey() {
  const userInput = DOM.userMsgInput || document.getElementById("userMsg");
  userInput.addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', async function() {
  console.log("DOM Content Loaded - Initializing chat page");
  
  // Cache DOM elements
  DOM.logoutBtn = document.getElementById("logoutBtn");
  DOM.toggleDarkModeBtn = document.getElementById("toggleDarkModeBtn");
  DOM.advanceTimeBtn = document.getElementById("advanceTimeBtn");
  DOM.newGameBtn = document.getElementById("newGameBtn");
  DOM.nyxSpaceBtn = document.getElementById("nyxSpaceBtn");
  DOM.convListDiv = document.getElementById("convList");
  DOM.chatWindow = document.getElementById("chatWindow");
  DOM.loadMoreBtn = document.getElementById("loadMore");
  DOM.userMsgInput = document.getElementById("userMsg");
  DOM.sendBtn = document.getElementById("sendBtn");
  DOM.contextMenuDiv = document.getElementById("contextMenu");
  DOM.leftPanelInner = document.getElementById("leftPanelInner");

  // Check login status
  const isLoggedIn = await checkLoggedIn();
  if (!isLoggedIn) return;

  // Initialize UI
  attachEnterKey();
  loadDarkModeFromStorage();
  await loadConversations();

  // Initialize socket connection
  socketManager.initialize();

  // Attach event listeners
  if (DOM.logoutBtn) DOM.logoutBtn.addEventListener("click", logout);
  if (DOM.toggleDarkModeBtn) DOM.toggleDarkModeBtn.addEventListener("click", toggleDarkMode);
  if (DOM.advanceTimeBtn) DOM.advanceTimeBtn.addEventListener("click", advanceTime);
  if (DOM.newGameBtn) DOM.newGameBtn.addEventListener("click", startNewGame);
  if (DOM.nyxSpaceBtn) DOM.nyxSpaceBtn.addEventListener("click", () => selectConversation(AppState.NYX_SPACE_CONV_ID));
  if (DOM.loadMoreBtn) DOM.loadMoreBtn.addEventListener("click", loadPreviousMessages);
  if (DOM.sendBtn) DOM.sendBtn.addEventListener("click", sendMessage);

  // Hide context menu on global click
  document.addEventListener("click", (e) => {
    if (DOM.contextMenuDiv && !DOM.contextMenuDiv.contains(e.target)) {
      DOM.contextMenuDiv.style.display = "none";
    }
  });
  
  // Handle page visibility changes
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      console.log("Page is now visible, checking connection");
      if (socketManager.socket && !socketManager.socket.connected) {
        console.log("Reconnecting socket...");
        socketManager.socket.connect();
      }
    }
  });
  
  // Handle window focus
  window.addEventListener("focus", () => {
    console.log("Window focused, checking connection");
    if (socketManager.socket && !socketManager.socket.connected) {
      console.log("Reconnecting socket...");
      socketManager.socket.connect();
    }
  });
  
  // Handle online/offline
  window.addEventListener("online", () => {
    console.log("Browser online");
    if (socketManager.socket && !socketManager.socket.connected) {
      socketManager.socket.connect();
    }
  });
  
  window.addEventListener("offline", () => {
    console.log("Browser offline");
  });

  console.log("Chat page initialization complete");
});
