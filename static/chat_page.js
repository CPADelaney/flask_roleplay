// chat_page.js - Production-ready version with all optimizations
console.log("Chat page loaded!");

// ===== Utility Functions =====
function sanitizeAndRenderMarkdown(markdownText) {
  try {
    if (typeof marked === 'undefined') {
      console.warn('Marked library not loaded, returning plain text');
      return DOMPurify.sanitize(markdownText);
    }
    const renderedHTML = marked.parse(markdownText || '');
    return DOMPurify.sanitize(renderedHTML);
  } catch (e) {
    console.error('Error rendering markdown:', e);
    return DOMPurify.sanitize(markdownText || '');
  }
}

// Dynamic DOM helper to avoid stale references
function $(id) {
  return document.getElementById(id);
}

// Helper function to ensure IDs are integers
function ensureIntegerId(id) {
    if (typeof id === 'string' && id !== 'anonymous') {
        return parseInt(id, 10);
    }
    return id;
}

// Helper function to check if ID should be converted to integer
function shouldConvertToInt(id) {
    return /^\d+$/.test(String(id));
}

// Helper function to safely convert IDs
function safeConvertId(id) {
    if (shouldConvertToInt(id)) {
        return ensureIntegerId(id);
    }
    return id;
}

// Centralized fetch helper with consistent error handling
async function fetchJson(url, opts = {}) {
  const res = await fetch(url, { credentials: 'include', ...opts });
  let data = null;
  
  try {
    data = await res.json();
  } catch (e) {
    // Response might not be JSON
  }
  
  if (!res.ok) {
    const error = new Error(data?.error || `${res.status} ${res.statusText}`);
    error.status = res.status;
    error.data = data;
    throw error;
  }
  
  return data;
}

// Normalize conversation IDs to strings
function normalizeConvId(id) {
  return id == null ? null : String(id);
}

// Validate URL is safe (http/https/protocol-relative)
function isValidImageUrl(url) {
  return /^(https?:)?\/\//i.test(url);
}

// Check if user is near bottom of scroll
function isNearBottom(element, threshold = 200) {
  return element.scrollHeight - element.scrollTop - element.clientHeight < threshold;
}

// Debug logging helper
function debugLog(...args) {
  if (window.localStorage.getItem('debugSocket') === '1') {
    console.log(...args);
  }
}

// ===== Configuration =====
const CONFIG = {
  MESSAGES_PER_LOAD: 20,
  NYX_SPACE_CONV_ID: "__nyx_space__",
  HEARTBEAT_INTERVAL: 20000,
  JOIN_TIMEOUT: 8000,
  SEND_TIMEOUT: 15000,
  GAME_POLL_INTERVAL: 3000,
  GAME_POLL_MAX_ATTEMPTS: 60,
  SCROLL_THRESHOLD: 200 // px from bottom to auto-scroll
};

// ===== State Management =====
const AppState = {
  // User info
  userId: null,
  isAdmin: false,
  
  // Conversation state
  currentConvId: null,      // Always normalized to string
  currentRoomId: null,      // Always stored as string
  roomConnectedOnce: false,
  messagesOffset: 0,
  
  // UI state
  isDarkMode: true,  // Default to dark mode
  isCreatingGame: false,
  isSelectingConversation: false,
  isSendingMessage: false,
  
  // Connection state
  socket: null,
  isConnected: false,
  reconnectionInProgress: false,
  
  // Message streaming
  currentAssistantBubble: null,
  partialAssistantMarkdown: "",
  _rafScheduled: false,
  
  // Timers/Intervals
  heartbeatInterval: null,
  
  // Pending operations
  pendingUniversalUpdates: null
};

// Universal updates object factory
function createUniversalUpdates() {
  return {
    batch_id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
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

// Initialize pending updates
AppState.pendingUniversalUpdates = createUniversalUpdates();

function resetPendingUniversalUpdates() {
  AppState.pendingUniversalUpdates = createUniversalUpdates();
}

// ===== Robust Socket Connection =====
function createRobustSocketConnection(handlers = {}) {
  const {
    onConnect = () => {},
    onDisconnect = () => {},
    onReconnect = () => {},
    onReconnectFailed = () => {}
  } = handlers;

  // Get the user ID from the window object (set by the template)
  const userId = window.CURRENT_USER_ID;
  
  // ADD DEBUGGING
  console.log('Creating socket with userId:', userId, 'type:', typeof userId);
  
  // Pass userId as query parameter since auth object isn't working
  const socket = io({
    path: '/socket.io/',
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    reconnectionAttempts: 10,
    timeout: 20000,
    autoConnect: true,
    // Pass user_id as query parameter instead of auth
    query: {
      user_id: userId
    }
  });

  // ADD MORE DEBUGGING
  console.log('Socket created with auth:', socket.auth);
  // Connection events
  socket.on('connect', () => {
    console.log('Socket connected:', socket.id, 'with user_id:', userId);
    onConnect(socket, false);
  });

  socket.on('disconnect', (reason) => {
    console.log('Socket disconnected:', reason);
    onDisconnect(socket, reason);
  });

  socket.on('reconnect', (attemptNumber) => {
    console.log('Socket reconnected after', attemptNumber, 'attempts');
    onReconnect(socket, attemptNumber);
  });

  socket.on('reconnect_failed', () => {
    console.log('Socket reconnection failed');
    onReconnectFailed();
  });

  socket.on('connect_error', (error) => {
    console.error('Socket connection error:', error.message);
  });

  return socket;
}

// ===== Admin UI Functions =====
function initializeAdminUI() {
  // Set admin state
  AppState.isAdmin = window.IS_ADMIN || false;
  
  if (AppState.isAdmin) {
    document.body.classList.add('is-admin');
  }
  
  // Update button visibility
  updateAdminButtonVisibility();
}

function updateAdminButtonVisibility() {
  const nyxSpaceBtn = $("nyxSpaceBtn");
  const advanceTimeBtn = $("advanceTimeBtn");
  
  if (nyxSpaceBtn) {
    nyxSpaceBtn.style.display = AppState.isAdmin ? "block" : "none";
  }
  
  if (advanceTimeBtn) {
    advanceTimeBtn.style.display = AppState.isAdmin ? "" : "none";
  }
}

// ===== New Game Dropdown Functions =====
function toggleNewGameDropdown() {
  const dropdown = $('newGameDropdown');
  console.log("toggleNewGameDropdown called, dropdown:", dropdown);
  
  if (!dropdown) {
    console.error("Dropdown element not found!");
    return;
  }
  
  const currentDisplay = window.getComputedStyle(dropdown).display;
  console.log("Computed display:", currentDisplay);
  
  dropdown.style.display = currentDisplay === 'none' ? 'block' : 'none';
  console.log("New display:", dropdown.style.display);
}

// Show preset stories modal
window.showPresetStories = async function() {
  const dropdown = $('newGameDropdown');
  dropdown.style.display = 'none';
  
  try {
    // Fetch available preset stories
    const response = await fetch('/new_game/api/preset-stories');
    const data = await response.json();
    
    // Populate the modal
    const storiesList = $('presetStoriesList');
    storiesList.innerHTML = '';
    
    if (data.stories && data.stories.length > 0) {
      data.stories.forEach(story => {
        const storyCard = document.createElement('div');
        storyCard.style.cssText = 'border: 1px solid #555; padding: 15px; margin-bottom: 10px; border-radius: 5px; cursor: pointer; transition: background-color 0.3s; background-color: transparent;';
        storyCard.onmouseover = () => storyCard.style.backgroundColor = '#4a4a4a';
        storyCard.onmouseout = () => storyCard.style.backgroundColor = 'transparent';
        
        storyCard.innerHTML = `
          <h3 style="margin: 0 0 10px 0;">${story.name}</h3>
          <p style="margin: 0 0 5px 0; color: #aaa;"><strong>Theme:</strong> ${story.theme}</p>
          <p style="margin: 0; color: #f0f0f0;">${story.synopsis}</p>
          <p style="margin: 5px 0 0 0; color: #888; font-size: 0.9em;">Acts: ${story.num_acts}</p>
        `;
        
        storyCard.onclick = () => startPresetGame(story.id);
        storiesList.appendChild(storyCard);
      });
    } else {
      storiesList.innerHTML = '<p>No preset stories available.</p>';
    }
    
    // Show the modal
    $('presetStoryModal').style.display = 'block';
    
  } catch (error) {
    console.error('Error fetching preset stories:', error);
    alert('Failed to load preset stories. Please try again.');
  }
}

// Close preset story modal
window.closePresetStoryModal = function() {
  $('presetStoryModal').style.display = 'none';
}

// Start a game with a preset story
async function startPresetGame(storyId) {
  if (AppState.isCreatingGame) return;

  closePresetStoryModal();

  const newGameBtn = $("newGameBtn");
  // Debug version - add this temporarily to chat_page.js
  if (newGameBtn) {
      newGameBtn.addEventListener("click", function(e) {
          e.preventDefault();
          e.stopPropagation();
          
          const dropdown = $('newGameDropdown');
          if (dropdown) {
              // Toggle display directly
              if (dropdown.style.display === 'none' || !dropdown.style.display) {
                  dropdown.style.display = 'block';
              } else {
                  dropdown.style.display = 'none';
              }
          } else {
              console.error('Dropdown element not found!');
          }
      });
  }

  AppState.isCreatingGame = true;

  const chatWindow = $("chatWindow");
  const loadingDiv = document.createElement("div");
  loadingDiv.id = "newGameLoadingIndicator";
  loadingDiv.innerHTML = '<div style="text-align:center;padding:20px;font-style:italic;color:#888;">Initializing preset story world...</div>';
  chatWindow.appendChild(loadingDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  try {
    const resp = await fetch("/new_game/api/new-game/preset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ story_id: storyId })
    });

    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Failed to start preset game");

    const convId = data.conversation_id;
    if (!convId) throw new Error("No conversation_id returned");

    // Track the new conversation
    AppState.currentConvId = normalizeConvId(convId);
    AppState.roomConnectedOnce = false;
    AppState.messagesOffset = 0;

    loadingDiv.innerHTML = '<div style="text-align:center;padding:20px;font-style:italic;color:#888;">Creating your preset story world... This may take a minute...</div>';

    // Poll until ready
    const poll = await pollForGameReady(convId);
    if (poll.ready) {
      loadingDiv.remove();

      // Join socket room if needed
      await socketManager.joinRoom(convId);

      // Load messages & assets
      await loadMessages(convId, true);
      await checkForWelcomeImage(convId);
      await loadConversations();

      if (poll.opening_narrative) {
        appendMessage({ sender: "Nyx", content: poll.opening_narrative }, true);
      } else {
        appendMessage({
          sender: "system",
          content: `New game started! Welcome to ${poll.conversation_name || "your new world"}.`
        }, true);
      }
    } else {
      throw new Error(poll.error || "Game creation timed out");
    }

  } catch (err) {
    console.error("startPresetGame error:", err);
    loadingDiv.remove();
    appendMessage({
      sender: "system",
      content: `Error starting preset game: ${err.message}. Please try again.`
    }, true);
  } finally {
    AppState.isCreatingGame = false;
    if (newGameBtn) {
      newGameBtn.disabled = false;
      newGameBtn.textContent = "➕ New Game ▼";
    }
  }
}

// Start custom game (wrapper for existing startNewGame function)
window.startCustomGame = async function() {
  closePresetStoryModal();
  await startNewGame();
}

// Start new game function
window.startNewGame = async function() {
  const dropdown = $('newGameDropdown');
  dropdown.style.display = 'none';
  
  if (AppState.isCreatingGame) {
    console.log("Game creation already in progress");
    return;
  }

  const newGameBtn = $("newGameBtn");
  if (newGameBtn) {
    newGameBtn.disabled = true;
    newGameBtn.textContent = "Creating...";
  }

  AppState.isCreatingGame = true;

  const chatWindow = $("chatWindow");
  if (!chatWindow) {
    AppState.isCreatingGame = false;
    if (newGameBtn) {
      newGameBtn.disabled = false;
      newGameBtn.textContent = "➕ New Game ▼";
    }
    return;
  }
  
  const loadingDiv = document.createElement("div");
  loadingDiv.id = "newGameLoadingIndicator";
  loadingDiv.innerHTML = '<div style="text-align: center; padding: 20px; font-style: italic; color: #888;">Initializing new game world...</div>';
  chatWindow.appendChild(loadingDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  try {
    const data = await fetchJson("/start_new_game", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });

    const newConvNum = data.conversation_id;
    AppState.currentConvId = normalizeConvId(newConvNum);
    AppState.roomConnectedOnce = false;
    AppState.messagesOffset = 0;

    // Update loading message
    loadingDiv.innerHTML = '<div style="text-align: center; padding: 20px; font-style: italic; color: #888;">Creating your world... This may take a minute...</div>';

    // Poll for completion
    const pollResult = await pollForGameReady(data.conversation_id);
    
    if (pollResult.ready) {
      loadingDiv.remove();
      
      // Join only if we didn't get auto-joined
      if (AppState.currentRoomId !== String(newConvNum)) {
        await socketManager.joinRoom(newConvNum);
      }
      
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
    
    const existingLoadingDiv = $("newGameLoadingIndicator");
    if (existingLoadingDiv) existingLoadingDiv.remove();
    
    appendMessage({ 
      sender: "system", 
      content: `Error starting new game: ${err.message}. Please try again.` 
    }, true);
  } finally {
    AppState.isCreatingGame = false;
    if (newGameBtn) {
      newGameBtn.disabled = false;
      newGameBtn.textContent = "➕ New Game ▼";
    }
  }
}

// ===== Socket Management =====
class SocketManager {
  constructor() {
    this.socket = null;
    this.pendingJoinTarget = null;
    this.joinPromise = null;
    this.joinResolver = null;
    this.joinRejecter = null;
    this.joinTimeout = null;
  }

  initialize() {
    if (this.socket) {
      console.warn("Socket already initialized");
      return;
    }
  
    // ADD THIS DEBUG
    console.log('About to create socket, window.CURRENT_USER_ID:', window.CURRENT_USER_ID);
    console.log('createRobustSocketConnection is:', typeof createRobustSocketConnection, typeof window.createRobustSocketConnection);
  
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

  destroy() {
    // Clean up heartbeat interval
    if (AppState.heartbeatInterval) {
      clearInterval(AppState.heartbeatInterval);
      AppState.heartbeatInterval = null;
    }
    
    // Clean up join timeout
    if (this.joinTimeout) {
      clearTimeout(this.joinTimeout);
      this.joinTimeout = null;
    }
    
    // Clear all join-related state
    this.pendingJoinTarget = null;
    this.joinPromise = null;
    this.joinResolver = null;
    this.joinRejecter = null;
    
    if (this.socket) {
      this.socket.removeAllListeners();
      this.socket.disconnect();
      this.socket = null;
    }
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
      this.joinRoom(AppState.currentConvId).catch(err => {
        console.error("Failed to rejoin room on reconnect:", err);
      });
    }
  }

  handleDisconnect(socket, reason) {
    console.error("Socket disconnected:", reason);
    AppState.isConnected = false;
    AppState.currentRoomId = null;
    AppState.reconnectionInProgress = true;
    
    // Clear sending state to prevent stuck UI
    AppState.isSendingMessage = false;
    
    // Re-enable input
    const userInput = $("userMsg");
    if (userInput) {
      userInput.disabled = false;
    }
    
    // Hide processing indicator
    removeProcessingIndicator();
    
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
      this.joinRoom(AppState.currentConvId).catch(err => {
        console.error("Failed to rejoin room after reconnect:", err);
      });
    }
  }

  handleReconnectFailed() {
    console.error("Socket reconnection failed");
    AppState.isSendingMessage = false;
    
    // Re-enable input
    const userInput = $("userMsg");
    if (userInput) {
      userInput.disabled = false;
    }
    
    removeProcessingIndicator();
    
    appendMessage({ 
      sender: "system", 
      content: "Unable to reconnect. Please refresh the page." 
    }, true);
  }

  joinRoom(conversationId) {
    if (!this.socket || !this.socket.connected) {
      return Promise.reject(new Error("Socket not connected"));
    }

    const target = normalizeConvId(conversationId);
    
    // Already in the target room
    if (AppState.currentRoomId === target && !this.pendingJoinTarget) {
      return Promise.resolve();
    }
    
    // Already joining this room
    if (this.pendingJoinTarget === target && this.joinPromise) {
      return this.joinPromise;
    }

    // Clear any previous join timeout
    if (this.joinTimeout) {
      clearTimeout(this.joinTimeout);
      this.joinTimeout = null;
    }

    this.pendingJoinTarget = target;
    this.joinPromise = new Promise((resolve, reject) => {
      // Store resolver and rejecter
      this.joinResolver = (room) => {
        const roomStr = normalizeConvId(room);
        if (roomStr === this.pendingJoinTarget) {
          this.pendingJoinTarget = null;
          this.joinPromise = null;
          this.joinRejecter = null;
          if (this.joinTimeout) {
            clearTimeout(this.joinTimeout);
            this.joinTimeout = null;
          }
          resolve();
        } else {
          console.warn("Received joined for unexpected room:", room, "expected:", this.pendingJoinTarget);
        }
      };
      
      this.joinRejecter = reject;
      
      // Set timeout
      this.joinTimeout = setTimeout(() => {
        if (this.pendingJoinTarget === target) {
          this.pendingJoinTarget = null;
          this.joinPromise = null;
          this.joinResolver = null;
          this.joinRejecter = null;
          reject(new Error(`Join room ${target} timed out`));
        }
      }, CONFIG.JOIN_TIMEOUT);
    });

    console.log(`Joining room ${target}`);
    this.socket.emit('join', { conversation_id: safeConvertId(conversationId) });
    return this.joinPromise;
  }

  setupMessageHandlers() {
    // Room events
    this.socket.on("joined", (data) => {
      console.log("Joined room:", data.room);
      AppState.currentRoomId = normalizeConvId(data.room);
      
      // Resolve join promise if pending
      if (this.joinResolver) {
        this.joinResolver(data.room);
      }
      
      // Only show connection message on initial join
      if (!AppState.roomConnectedOnce) {
        appendMessage({ sender: "system", content: "Connected to game session" }, true);
        AppState.roomConnectedOnce = true;
      }
    });

    // Handle join errors
    this.socket.on("join_error", (error) => {
      console.error("Join error:", error);
      if (this.joinRejecter) {
        this.pendingJoinTarget = null;
        this.joinPromise = null;
        this.joinResolver = null;
        const rejecter = this.joinRejecter;
        this.joinRejecter = null;
        rejecter(new Error(error.message || 'Join failed'));
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
      
      // Reset universal updates only on success
      resetPendingUniversalUpdates();
    });

    this.socket.on("error", (payload) => {
      console.error("Server error:", payload.error);
      removeProcessingIndicator();
      appendMessage({ 
        sender: "system", 
        content: `Error: ${payload.error}` 
      }, true);
      AppState.isSendingMessage = false;
      
      // Reset universal updates on error too
      resetPendingUniversalUpdates();
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
      debugLog("Received server heartbeat:", data.timestamp);
    });
  }

  setupHeartbeat() {
    // Clear any existing interval
    if (AppState.heartbeatInterval) {
      clearInterval(AppState.heartbeatInterval);
    }
    
    AppState.heartbeatInterval = setInterval(() => {
      // Early return if disconnected
      if (!this.socket || !this.socket.connected) {
        return;
      }
      this.socket.emit('client_heartbeat', { timestamp: Date.now() });
    }, CONFIG.HEARTBEAT_INTERVAL);
  }

  async sendMessage(data) {
    if (!this.socket || !this.socket.connected) {
      console.error("Cannot send message - socket not connected");
      return false;
    }

    // Ensure conversation_id is properly formatted
    if (data.conversation_id !== undefined) {
      data.conversation_id = safeConvertId(data.conversation_id);
    }

    const roomIdStr = normalizeConvId(data.conversation_id);
    if (AppState.currentRoomId !== roomIdStr) {
      console.warn("Not in the correct room, joining first");
      try {
        await this.joinRoom(data.conversation_id);
      } catch (err) {
        console.error("Failed to join room:", err);
        return false;
      }
    }

    this.socket.emit("storybeat", data);
    return true;
  }
}

// Create global socket manager instance
const socketManager = new SocketManager();

// ===== Message Display Functions =====
const MessageFactory = {
  create(message, autoScroll = true) {
    const chatWindow = $("chatWindow");
    if (!chatWindow) return null;
    
    // Check if we should auto-scroll before adding message
    const shouldScroll = autoScroll && isNearBottom(chatWindow, CONFIG.SCROLL_THRESHOLD);
    
    const bubbleRow = this.createBubble(message);
    chatWindow.appendChild(bubbleRow);
    
    if (shouldScroll) {
      requestAnimationFrame(() => {
        chatWindow.scrollTop = chatWindow.scrollHeight;
      });
    }
    
    return bubbleRow;
  },

  createBubble(message) {
    const row = document.createElement("div");
    row.classList.add("message-row");
    
    const isUser = message.sender && message.sender.toLowerCase() === "user";
    row.classList.add(isUser ? "user-row" : "gpt-row");
    
    const bubble = document.createElement("div");
    bubble.classList.add("message-bubble");
    
    // Create elements instead of using innerHTML
    const senderStrong = document.createElement("strong");
    senderStrong.textContent = message.sender + ":";
    bubble.appendChild(senderStrong);
    
    const contentSpan = document.createElement("span");
    contentSpan.innerHTML = " " + sanitizeAndRenderMarkdown(message.content || "");
    bubble.appendChild(contentSpan);
    
    row.appendChild(bubble);
    return row;
  },

  createImage(imageUrl, reason) {
    const chatWindow = $("chatWindow");
    if (!chatWindow) return;
    
    // Validate URL safety
    if (!isValidImageUrl(imageUrl)) {
      console.error("Invalid image URL:", imageUrl);
      return;
    }
    
    // Check if we should auto-scroll before adding image
    const shouldScroll = isNearBottom(chatWindow, CONFIG.SCROLL_THRESHOLD);
    
    const row = document.createElement("div");
    row.className = "message-row gpt-row";
    
    const bubble = document.createElement("div");
    bubble.className = "message-bubble image-bubble";

    const container = document.createElement("div");
    container.className = "image-container";

    const img = document.createElement("img");
    img.src = imageUrl; // Browser handles URL safely
    img.alt = "Generated scene";
    img.style.maxWidth = "100%";
    img.style.borderRadius = "5px";
    img.loading = "lazy"; // Performance optimization

    const caption = document.createElement("div");
    caption.className = "image-caption";
    caption.textContent = reason || "AI-generated scene visualization";

    container.appendChild(img);
    container.appendChild(caption);
    bubble.appendChild(container);
    row.appendChild(bubble);
    
    chatWindow.appendChild(row);
    
    if (shouldScroll) {
      requestAnimationFrame(() => {
        chatWindow.scrollTop = chatWindow.scrollHeight;
      });
    }
  }
};

// Wrapper functions for compatibility
function appendMessage(message, autoScroll = true) {
  return MessageFactory.create(message, autoScroll);
}

function createBubble(message) {
  return MessageFactory.createBubble(message);
}

function appendImageToChat(imageUrl, reason) {
  return MessageFactory.createImage(imageUrl, reason);
}

// Optimized token handling with throttling
function handleNewToken(token) {
  removeProcessingIndicator();
  const chatWindow = $("chatWindow");
  if (!chatWindow) return;

  if (!AppState.currentAssistantBubble) {
    const row = document.createElement("div");
    row.classList.add("message-row", "gpt-row");
    
    const bubble = document.createElement("div");
    bubble.classList.add("message-bubble");
    
    const senderStrong = document.createElement("strong");
    senderStrong.textContent = "Nyx: ";
    bubble.appendChild(senderStrong);
    
    const contentSpan = document.createElement('span');
    bubble.appendChild(contentSpan);

    row.appendChild(bubble);
    chatWindow.appendChild(row);
    
    AppState.currentAssistantBubble = contentSpan;
    AppState.partialAssistantMarkdown = "";
  }
  
  // Accumulate markdown
  AppState.partialAssistantMarkdown += token;
  
  // Simplified throttled UI update
  if (!AppState._rafScheduled) {
    AppState._rafScheduled = true;
    
    requestAnimationFrame(() => {
      if (AppState.currentAssistantBubble) {
        AppState.currentAssistantBubble.textContent = AppState.partialAssistantMarkdown;
        
        // Only auto-scroll if user is near bottom
        if (isNearBottom(chatWindow, CONFIG.SCROLL_THRESHOLD)) {
          chatWindow.scrollTop = chatWindow.scrollHeight;
        }
      }
      AppState._rafScheduled = false;
    });
  }
}

function finalizeAssistantMessage(finalText) {
  // Use server text if valid, otherwise fall back to accumulated markdown
  const text = (typeof finalText === 'string' && finalText.trim().length > 0)
    ? finalText
    : AppState.partialAssistantMarkdown;

  if (!AppState.currentAssistantBubble) {
    if (text && text.trim()) {
      appendMessage({ sender: "Nyx", content: text }, true);
    }
  } else {
    // Parse markdown once at the end
    try {
      AppState.currentAssistantBubble.innerHTML = sanitizeAndRenderMarkdown(text);
    } catch (e) {
      console.error('Error finalizing message:', e);
      AppState.currentAssistantBubble.textContent = text;
    }
  }
  
  AppState.currentAssistantBubble = null;
  AppState.partialAssistantMarkdown = "";
  
  const chatWindow = $("chatWindow");
  if (chatWindow) {
    requestAnimationFrame(() => {
      chatWindow.scrollTop = chatWindow.scrollHeight;
    });
  }
}

function showProcessingIndicator() {
  const chatWindow = $("chatWindow");
  if (!chatWindow) return;
  
  removeProcessingIndicator();
  
  const processingDiv = document.createElement("div");
  processingDiv.id = "processingIndicator";
  processingDiv.innerHTML = '<div style="text-align: center; padding: 10px; font-style: italic; color: #888;">Processing your request...</div>';
  chatWindow.appendChild(processingDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function removeProcessingIndicator() {
  const indicator = $("processingIndicator");
  if (indicator) {
    indicator.remove();
  }
}

// ===== User Actions =====
async function sendMessage() {
  const userInput = $("userMsg");
  if (!userInput) return;
  
  const userText = userInput.value.trim();
  
  if (!userText || !AppState.currentConvId) {
    return;
  }

  // Prevent double sending
  if (AppState.isSendingMessage) {
    console.warn("Already sending a message");
    return;
  }

  // Set sending state immediately
  AppState.isSendingMessage = true;
  userInput.value = "";
  userInput.disabled = true; // Prevent input during send

  try {
    // Handle Nyx Space differently
    if (AppState.currentConvId === CONFIG.NYX_SPACE_CONV_ID) {
      try {
        await handleNyxSpaceMessage(userText);
      } catch (nyxError) {
        console.error("Error in Nyx Space handler:", nyxError);
        appendMessage({ 
          sender: "system", 
          content: "Failed to process Nyx Space message." 
        }, true);
      }
      return;
    }

    // Display user message
    appendMessage({ sender: "user", content: userText }, true);

    // Reset streaming state
    AppState.currentAssistantBubble = null;
    AppState.partialAssistantMarkdown = "";

    // Prepare message data
    const messageData = {
      user_input: userText,
      conversation_id: safeConvertId(AppState.currentConvId),
      player_name: "Chase",
      advance_time: false,
      universal_update: AppState.pendingUniversalUpdates
    };

    // Send message
    console.log(`Sending message for conversation ${AppState.currentConvId}`);
    
    const sent = await socketManager.sendMessage(messageData);
    if (!sent) {
      throw new Error("Failed to send message");
    }

    // Set timeout for response
    setTimeout(() => {
      if (AppState.isSendingMessage) {
        appendMessage({ 
          sender: "system", 
          content: "Server is taking longer than expected. Please wait..." 
        }, true);
        // Clear stale data on timeout
        resetPendingUniversalUpdates();
        AppState.isSendingMessage = false;
      }
    }, CONFIG.SEND_TIMEOUT);
    
  } catch (err) {
    console.error("Error sending message:", err);
    appendMessage({ 
      sender: "system", 
      content: "Failed to send message. Please check your connection." 
    }, true);
    AppState.isSendingMessage = false;
  } finally {
    userInput.disabled = false;
    // Ensure flag is cleared even if Nyx handler fails
    if (AppState.currentConvId === CONFIG.NYX_SPACE_CONV_ID) {
      AppState.isSendingMessage = false;
    }
  }
}

async function handleNyxSpaceMessage(userText) {
  try {
    // Save user message
    await fetchJson('/nyx_space/messages', {
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
    const replyData = await fetchJson('/admin/nyx_direct', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        user_input: userText,
        generate_response: true,
        conversation_id: safeConvertId(AppState.currentConvId)
      })
    });

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
    await fetchJson('/nyx_space/messages', {
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
  } finally {
    AppState.isSendingMessage = false;
  }
}

async function advanceTime() {
  if (!AppState.currentConvId) {
    alert("Please select a conversation first");
    return;
  }

  // Prevent double sending
  if (AppState.isSendingMessage) {
    console.warn("Already processing a message");
    return;
  }

  AppState.isSendingMessage = true;

  appendMessage({ 
    sender: "user", 
    content: "Let's advance to the next time period." 
  }, true);

  // Reset streaming state
  AppState.currentAssistantBubble = null;
  AppState.partialAssistantMarkdown = "";

  const messageData = {
    user_input: "Let's advance to the next time period.",
    conversation_id: safeConvertId(AppState.currentConvId),
    player_name: "Chase",
    advance_time: true,
    universal_update: AppState.pendingUniversalUpdates
  };

  try {
    const sent = await socketManager.sendMessage(messageData);
    if (!sent) {
      throw new Error("Failed to advance time");
    }
  } catch (err) {
    console.error("Error advancing time:", err);
    appendMessage({ 
      sender: "system", 
      content: "Failed to advance time. Please check your connection." 
    }, true);
    AppState.isSendingMessage = false;
  }
}

// ===== Conversation Management =====
window.selectConversation = async function(convId) {
  if (AppState.isSelectingConversation) {
    console.log("Already selecting a conversation");
    return;
  }

  AppState.roomConnectedOnce = false;   // reset flag when switching rooms
  AppState.isSelectingConversation = true;
  AppState.currentConvId = normalizeConvId(convId);
  AppState.messagesOffset = 0;
  
  // Reset pending updates when switching conversations
  resetPendingUniversalUpdates();

  console.log(`Selecting conversation: ${AppState.currentConvId}`);

  // Join the room
  if (socketManager.socket && socketManager.socket.connected) {
    try {
      await socketManager.joinRoom(convId); // Pass original value for special IDs like "__nyx_space__"
    } catch (err) {
      console.error("Failed to join conversation room:", err);
      appendMessage({ 
        sender: "system", 
        content: "Failed to join conversation. Please try again." 
      }, true);
    }
  }

  // Handle different conversation types
  if (AppState.currentConvId === CONFIG.NYX_SPACE_CONV_ID) {
    await loadNyxSpace();
  } else {
    await loadGameConversation(convId); // Pass original convId
  }

  // Update admin button visibility for game conversations
  if (AppState.currentConvId !== CONFIG.NYX_SPACE_CONV_ID) {
    const advanceTimeBtn = $("advanceTimeBtn");
    if (advanceTimeBtn) {
      advanceTimeBtn.style.display = AppState.isAdmin ? "" : "none";
    }
  }

  AppState.isSelectingConversation = false;
}

async function loadNyxSpace() {
  const chatWindow = $("chatWindow");
  if (!chatWindow) return;
  
  chatWindow.innerHTML = "";
  
  appendMessage({sender: "system", content: "Loading Nyx's Space..."}, true);
  
  try {
    const data = await fetchJson("/nyx_space/messages");
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
  } catch (err) {
    console.error("Error loading Nyx Space:", err);
    chatWindow.innerHTML = "";
    appendMessage({
      sender: "Nyx", 
      content: "There was an error loading Nyx's Space."
    }, true);
  }
  
  // Hide game-specific buttons
  const advanceTimeBtn = $("advanceTimeBtn");
  const loadMoreBtn = $("loadMore");
  if (advanceTimeBtn) advanceTimeBtn.style.display = "none";
  if (loadMoreBtn) loadMoreBtn.style.display = "none";
}

async function loadGameConversation(convId) {
  // Show game-specific buttons
  const advanceTimeBtn = $("advanceTimeBtn");
  const loadMoreBtn = $("loadMore");
  if (advanceTimeBtn) advanceTimeBtn.style.display = AppState.isAdmin ? "" : "none";
  if (loadMoreBtn) loadMoreBtn.style.display = "";
  
  await loadMessages(convId, true);
  await checkForWelcomeImage(convId);
}

// ===== Conversation messages loader =====
async function loadMessages(convId, replace = false) {
  const chatWindow  = $("chatWindow");
  let   loadMoreBtn = $("loadMore");
  if (!chatWindow) return;

  const url = `/multiuser/conversations/${ensureIntegerId(convId)}/messages` +
              `?offset=${AppState.messagesOffset}&limit=${CONFIG.MESSAGES_PER_LOAD}`;

  try {
    const data = await fetchJson(url);

    // ---------- Hard replace branch ----------
    if (replace) {
      // 1️⃣ remove the existing Load-More span from DOM (if any)
      if (loadMoreBtn && loadMoreBtn.parentNode) {
        loadMoreBtn.remove();
      }

      // 2️⃣ clear ALL previous message rows
      chatWindow.innerHTML = "";

      // 3️⃣ (re)create the Load-More span
      const btn = loadMoreBtn || document.createElement("span");
      btn.id = "loadMore";
      btn.textContent = "Load older messages...";
      btn.style.display = "none";
      btn.addEventListener("click", loadPreviousMessages);

      chatWindow.appendChild(btn);

      // 4️⃣ keep our reference up-to-date for the rest of this function
      if (!loadMoreBtn) loadMoreBtn = btn;
    }
    // ---------- end replace ----------

    // Preserve scroll position if we're prepending messages
    const prevHeight = chatWindow.scrollHeight;

    // Build DOM nodes for the fetched messages
    const frag = document.createDocumentFragment();
    data.messages
        .slice()         // shallow copy
        .reverse()       // oldest first
        .forEach(msg => frag.appendChild(createBubble(msg)));

    if (replace) {
      chatWindow.appendChild(frag);
      chatWindow.scrollTop = chatWindow.scrollHeight;   // scroll to bottom
    } else {
      loadMoreBtn.after(frag);                          // prepend
      chatWindow.scrollTop += chatWindow.scrollHeight - prevHeight;
    }

    // toggle visibility of the Load-More control
    loadMoreBtn.style.display =
      data.messages.length < CONFIG.MESSAGES_PER_LOAD ? "none" : "block";

  } catch (err) {
    console.error("Error loading messages:", err);
    appendMessage({ sender: "system", content: "Error loading messages." }, true);
  }
}

async function checkForWelcomeImage(convId) {
  try {
    const data = await fetchJson(`/universal/get_roleplay_value?conversation_id=${ensureIntegerId(convId)}&key=WelcomeImageUrl`);
    if (data.value) {
      appendImageToChat(data.value, "Welcome to this new world");
    }
  } catch (err) {
    console.error("Error checking for welcome image:", err);
  }
}

async function pollForGameReady(conversationId) {
  const maxAttempts = CONFIG.GAME_POLL_MAX_ATTEMPTS;
  let attempts = 0;

  while (attempts < maxAttempts) {
    attempts++;
    
    try {
      const statusData = await fetchJson(`/new_game/conversation_status?conversation_id=${ensureIntegerId(conversationId)}`);
      
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
    } catch (pollError) {
      console.error("Error polling game status:", pollError);
    }
    
    // Wait before next attempt
    await new Promise(resolve => setTimeout(resolve, CONFIG.GAME_POLL_INTERVAL));
  }
  
  return {
    ready: false,
    error: "Game creation timed out"
  };
}

async function loadConversations() {
  try {
    const convoData = await fetchJson("/multiuser/conversations");
    renderConvoList(convoData);
  } catch (err) {
    console.error("Error loading conversations:", err);
  }
}

function renderConvoList(conversations) {
  const convListDiv = $("convList");
  if (!convListDiv) return;
  
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
let contextMenuConvId = null;

function showContextMenu(x, y, convId) {
  const menuDiv = $("contextMenu");
  if (!menuDiv) return;
  
  // Clear any existing menu state
  menuDiv.style.display = "none";
  contextMenuConvId = convId;
  
  // Clear old content and listeners
  menuDiv.replaceChildren();

  const options = [
    { text: "Rename", action: "rename" },
    { text: "Move to Folder", action: "move" },
    { text: "Delete", action: "delete" }
  ];

  options.forEach(opt => {
    const div = document.createElement("div");
    div.textContent = opt.text;
    div.dataset.action = opt.action;
    menuDiv.appendChild(div);
  });

  menuDiv.style.left = x + "px";
  menuDiv.style.top = y + "px";
  
  // Show menu after a tick to avoid immediate close
  setTimeout(() => {
    menuDiv.style.display = "block";
  }, 0);
}

// Single event handler for context menu
function handleContextMenuClick(e) {
  const menuDiv = $("contextMenu");
  if (!menuDiv || !menuDiv.contains(e.target)) return;
  
  const action = e.target.dataset.action;
  if (!action || !contextMenuConvId) return;
  
  e.stopPropagation();
  menuDiv.style.display = "none";
  
  switch(action) {
    case "rename":
      renameConversation(contextMenuConvId);
      break;
    case "move":
      moveConversationToFolder(contextMenuConvId);
      break;
    case "delete":
      deleteConversation(contextMenuConvId);
      break;
  }
  
  contextMenuConvId = null;
}

async function renameConversation(convId) {
  const newName = prompt("Enter new conversation name:");
  if (!newName || newName.trim() === "") return;
  
  try {
    await fetchJson(`/multiuser/conversations/${ensureIntegerId(convId)}`, {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ conversation_name: newName.trim() })
    });
    
    await loadConversations();
  } catch (err) {
    alert(`Error renaming conversation: ${err.message}`);
  }
}

async function moveConversationToFolder(convId) {
  const folderName = prompt("Enter folder name:");
  if (!folderName || folderName.trim() === "") return;
  
  try {
    await fetchJson(`/multiuser/conversations/${ensureIntegerId(convId)}/move_folder`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder_name: folderName.trim() })
    });
    
    await loadConversations();
  } catch (err) {
    alert(`Error moving conversation: ${err.message}`);
  }
}

async function deleteConversation(convId) {
  if (!confirm("Are you sure you want to delete this conversation? This cannot be undone.")) {
    return;
  }
  
  try {
    await fetchJson(`/multiuser/conversations/${ensureIntegerId(convId)}`, {
      method: "DELETE"
    });
    
    await loadConversations();
    
    if (normalizeConvId(AppState.currentConvId) === normalizeConvId(convId)) {
      const chatWindow = $("chatWindow");
      if (chatWindow) {
        // Reuse existing load more button or create new one
        let loadMoreBtn = $("loadMore");
        if (!loadMoreBtn) {
          loadMoreBtn = document.createElement("span");
          loadMoreBtn.id = "loadMore";
          loadMoreBtn.textContent = "Load older messages...";
          loadMoreBtn.style.display = "none";
          loadMoreBtn.addEventListener("click", loadPreviousMessages);
        }
        chatWindow.innerHTML = "";
        chatWindow.appendChild(loadMoreBtn);
      }
      AppState.currentConvId = null;
      AppState.currentRoomId = null;
    }
  } catch (err) {
    alert(`Error deleting conversation: ${err.message}`);
  }
}

// ===== Utility Functions =====
async function checkLoggedIn() {
  try {
    const data = await fetchJson("/whoami");
    
    if (!data.logged_in || !data.user_id || data.user_id === "anonymous") {
      window.location.href = "/login_page";
      return false;
    }
    
    // Update the global state with the valid user ID
    AppState.userId = data.user_id;
    window.CURRENT_USER_ID = data.user_id; // Ensure consistency
    
    const logoutBtn = $("logoutBtn");
    if (logoutBtn) {
      logoutBtn.style.display = "inline-block";
    }
    
    return true;
  } catch (err) {
    console.error("Error checking login status:", err);
    window.location.href = "/login_page";
    return false;
  }
}

function loadThemeFromStorage() {
  const savedTheme = localStorage.getItem("theme");
  
  // Default to dark mode
  if (!savedTheme || savedTheme === "dark") {
    AppState.isDarkMode = true;
    document.body.classList.remove("light-mode");
  } else {
    AppState.isDarkMode = false;
    document.body.classList.add("light-mode");
  }
}

function toggleTheme() {
  AppState.isDarkMode = !AppState.isDarkMode;
  
  if (AppState.isDarkMode) {
    document.body.classList.remove("light-mode");
    localStorage.setItem("theme", "dark");
  } else {
    document.body.classList.add("light-mode");
    localStorage.setItem("theme", "light");
  }
  
  // Update button text if it exists
  const themeBtn = $("toggleThemeBtn");
  if (themeBtn) {
    themeBtn.textContent = AppState.isDarkMode ? "☀️ Light Mode" : "🌙 Dark Mode";
  }
}

async function logout() {
  try {
    await fetchJson("/logout", { method: "POST" });
    window.location.href = "/login_page";
  } catch (err) {
    console.error("Logout error:", err);
    alert("Logout failed!");
  }
}

function loadPreviousMessages() {
  if (!AppState.currentConvId) return;
  AppState.messagesOffset += CONFIG.MESSAGES_PER_LOAD;
  loadMessages(AppState.currentConvId, false);
}

function attachEnterKey() {
  const userInput = $("userMsg");
  if (!userInput) return;
  
  userInput.addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}

// Keyboard accessibility for context menu
function handleKeyboard(e) {
  const contextMenu = $("contextMenu");
  if (!contextMenu || contextMenu.style.display === "none") return;
  
  if (e.key === "Escape") {
    contextMenu.style.display = "none";
    contextMenuConvId = null;
  }
}

// Global unhandled rejection handler
window.addEventListener("unhandledrejection", e => {
  console.error("Unhandled promise rejection:", e.reason);
  
  let message = "An error occurred";
  if (e.reason instanceof Error) {
    if (e.reason.message.includes("Failed to fetch") || e.reason.message.includes("NetworkError")) {
      message = "Network error. Please check your connection.";
    } else {
      message = `Error: ${e.reason.message}`;
    }
  }
  
  appendMessage({
    sender: "system", 
    content: message
  }, true);
  
  e.preventDefault();
});

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', async function() {
  console.log("DOM Content Loaded - Initializing chat page");
  
  // Check if we have a valid user ID BEFORE doing anything else
  if (!window.CURRENT_USER_ID || window.CURRENT_USER_ID === "anonymous") {
    console.error("No valid user ID found, redirecting to login");
    window.location.href = "/login_page";
    return; // Stop all initialization
  }
  
  // Check login status via API
  const isLoggedIn = await checkLoggedIn();
  if (!isLoggedIn) {
    // checkLoggedIn already redirects, but ensure we stop execution
    return;
  }

  // Only initialize socket after confirming authentication
  console.log("User authenticated, initializing application...");
  
  // Initialize admin UI
  initializeAdminUI();

  // Initialize UI
  attachEnterKey();
  loadThemeFromStorage();
  await loadConversations();

  // Initialize socket connection
  socketManager.initialize();

  // Attach event listeners using dynamic lookups
  const logoutBtn = $("logoutBtn");
  const toggleThemeBtn = $("toggleThemeBtn");
  const advanceTimeBtn = $("advanceTimeBtn");
  const newGameBtn = $("newGameBtn");
  const nyxSpaceBtn = $("nyxSpaceBtn");
  const loadMoreBtn = $("loadMore");
  const sendBtn = $("sendBtn");

  if (logoutBtn) logoutBtn.addEventListener("click", logout);
  if (toggleThemeBtn) {
    toggleThemeBtn.addEventListener("click", toggleTheme);
    // Set initial button text
    toggleThemeBtn.textContent = AppState.isDarkMode ? "☀️ Light Mode" : "🌙 Dark Mode";
  }
  if (advanceTimeBtn) advanceTimeBtn.addEventListener("click", advanceTime);
  if (newGameBtn) {
      newGameBtn.addEventListener("click", function(e) {
          e.stopPropagation(); // Prevent the click from bubbling up
          toggleNewGameDropdown();
      });
  }
  if (nyxSpaceBtn) nyxSpaceBtn.addEventListener("click", () => selectConversation(CONFIG.NYX_SPACE_CONV_ID));
  if (loadMoreBtn) loadMoreBtn.addEventListener("click", loadPreviousMessages);
  if (sendBtn) sendBtn.addEventListener("click", sendMessage);

  // Context menu handlers
  document.addEventListener("click", (e) => {
      const contextMenu = $("contextMenu");
      const dropdown = $('newGameDropdown');
      const newGameBtn = $('newGameBtn');
      
      // Don't close dropdown if clicking the button or dropdown itself
      if (newGameBtn && (newGameBtn.contains(e.target) || dropdown?.contains(e.target))) {
          // Don't close the dropdown
          if (contextMenu) {
              contextMenu.style.display = "none";
              contextMenuConvId = null;
          }
          return;
      }
      
      // Close context menu if clicking outside
      if (contextMenu && !contextMenu.contains(e.target)) {
          contextMenu.style.display = "none";
          contextMenuConvId = null;
      } else if (contextMenu && contextMenu.contains(e.target)) {
          handleContextMenuClick(e);
      }
      
      // Close dropdown if clicking outside
      if (dropdown && dropdown.style.display === 'block') {
          dropdown.style.display = 'none';
      }
  });
  
  // Keyboard accessibility
  document.addEventListener("keydown", handleKeyboard);
  
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
    debugLog("Window focused, checking connection");
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

  // Cleanup on page unload
  window.addEventListener("beforeunload", () => {
    if (socketManager) {
      socketManager.destroy();
    }
  });

  console.log("Chat page initialization complete");
});
