// chat_page.js
console.log("Main JS file loaded!");

// Utility: Markdown + sanitization
function sanitizeAndRenderMarkdown(markdownText) {
  const renderedHTML = marked.parse(markdownText);
  return DOMPurify.sanitize(renderedHTML);
}

// Global conversation info
let currentConvId = null;
let messagesOffset = 0;
const MESSAGES_PER_LOAD = 20;
let isDarkMode = false;
let isCreatingGame = false;
let isSelectingConversation = false;

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

// Socket.IO reference
let socket = null;
let reconnectionInProgress = false;

// We accumulate the partial streamed content in a single bubble
let currentAssistantBubble = null;
let partialAssistantText = "";

// DOM Elements (cache them after DOMContentLoaded)
let logoutBtn, toggleDarkModeBtn, advanceTimeBtn, newGameBtn, convListDiv,
    chatWindow, loadMoreBtn, userMsgInput, sendBtn, contextMenuDiv, leftPanelInner;


function setupSocketListeners() {
  // Connection events
  socket.on("connect", () => {
    console.log("Socket.IO connected with ID:", socket.id);
    
    // If reconnection was in progress, add a system message
    if (reconnectionInProgress) {
      const reconnectMsg = { sender: "system", content: "Connection restored!" };
      appendMessage(reconnectMsg, true);
      reconnectionInProgress = false;
    }
    
    // Join conversation room if one is active
    if (currentConvId) {
      socket.emit('join', { conversation_id: currentConvId });
      console.log(`Joined room: ${currentConvId}`);
    }
  });

  socket.on("disconnect", (reason) => {
    console.error("Socket.IO disconnected DETAILED:", reason, "Current socket ID:", socket.id, "Connected:", socket.connected); // More detail
    reconnectionInProgress = true;
    const disconnectMsg = { sender: "system", content: `Connection lost (${reason}). Attempting to reconnect...` };
    appendMessage(disconnectMsg, true);
    
    // Server-initiated disconnect needs manual reconnection
    if (reason === 'io server disconnect') {
      setTimeout(() => socket.connect(), 1000);
    }
  });

  socket.on("connect_error", (error) => {
    console.error("Socket.IO connection error:", error);
  });
  
  socket.on("reconnect_attempt", (attemptNumber) => {
    console.log(`Socket.IO reconnection attempt #${attemptNumber}`);
    
    // After several reconnection attempts, try alternate transports
    if (attemptNumber % 3 === 0) {
      console.log("Trying alternate transport strategy...");
    }
  });
  
  socket.on("reconnect", (attemptNumber) => {
    console.log(`Socket.IO reconnected after ${attemptNumber} attempts`);
    
    // Reset reconnection state
    reconnectionInProgress = false;
    
    // Rejoin the conversation room if active
    if (currentConvId) {
      socket.emit('join', { conversation_id: currentConvId });
    }
  });
  
  socket.on("reconnect_failed", () => {
    console.error("Socket.IO reconnection failed");
    const failMsg = { sender: "system", content: "Unable to reconnect. Please refresh the page." };
    appendMessage(failMsg, true);
  });

  // Room events
  socket.on("joined", (data) => {
    console.log("Joined room:", data.room);
    const joinMsg = { sender: "system", content: "Connected to game session" };
    appendMessage(joinMsg, true);
  });

  // Message streaming events
  socket.on("new_token", function(payload) {
    handleNewToken(payload.token);
  });

  socket.on("done", function(payload) {
    console.log("Done streaming. Full text received.");
    finalizeAssistantMessage(payload.full_text);
  });

  socket.on("error", (payload) => {
    console.error("Server error:", payload.error);
    handleNewToken("[Error: " + payload.error + "]");
    finalizeAssistantMessage(""); // Finalize with empty if error occurred during stream
  });

  socket.on("message", function(data) {
    console.log("Received non-streaming message event:", data);
    const messageObj = {
      sender: data.sender || "Nyx",
      content: data.content || data.message || "No content"
    };
    appendMessage(messageObj, true);
  });

  socket.on("image", function(payload) {
     console.log("Received image:", payload);
     appendImageToChat(payload.image_url, payload.reason);
   });

  socket.on("processing", function(data) {
     console.log("Server is processing:", data.message);
     const chatWindowEl = document.getElementById("chatWindow"); // Use direct ID if needed before cache
     let processingDiv = document.getElementById("processingIndicator");
     if (!processingDiv) {
         processingDiv = document.createElement("div");
         processingDiv.id = "processingIndicator";
         processingDiv.innerHTML = '<div style="text-align: center; padding: 10px; font-style: italic; color: #888;">Processing your request...</div>';
         chatWindowEl.appendChild(processingDiv);
     }
     chatWindowEl.scrollTop = chatWindowEl.scrollHeight;
  });

  socket.on("game_state_update", function(data) {
     console.log("Game state updated:", data);
     // Example: if (data.type === "npc_update") updateNPCInfo(data.npc_data);
  });
  
  // Heartbeat handler to keep connection alive
  socket.on("server_heartbeat", function(data) {
    console.log("Received server heartbeat:", data.timestamp);
  });
  
  // Set up client heartbeat to keep connection alive
  setInterval(() => {
    if (socket && socket.connected) {
      socket.emit('client_heartbeat', { timestamp: Date.now() });
    }
  }, 20000); // Send heartbeat every 20 seconds
}

 function appendImageToChat(imageUrl, reason) {
     const chatWindowEl = chatWindow || document.getElementById("chatWindow");
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
     chatWindowEl.appendChild(imageRow);
     chatWindowEl.scrollTop = chatWindowEl.scrollHeight;
 }


function handleNewToken(token) {
  const processingIndicator = document.getElementById("processingIndicator");
  if (processingIndicator) {
    processingIndicator.remove();
  }
  const chatWindowEl = chatWindow || document.getElementById("chatWindow");

  if (!currentAssistantBubble) {
    const row = document.createElement("div");
    row.classList.add("message-row", "gpt-row");
    const bubble = document.createElement("div");
    bubble.classList.add("message-bubble");
    bubble.innerHTML = `<strong>Nyx:</strong> `; // Start with sender
    const contentSpan = document.createElement('span'); // Create a span for the actual content
    contentSpan.innerHTML = sanitizeAndRenderMarkdown(token);
    bubble.appendChild(contentSpan);

    row.appendChild(bubble);
    chatWindowEl.appendChild(row);
    currentAssistantBubble = contentSpan; // The bubble to append to is now the span
    partialAssistantText = token;
  } else {
    partialAssistantText += token;
    currentAssistantBubble.innerHTML = sanitizeAndRenderMarkdown(partialAssistantText);
  }
  chatWindowEl.scrollTop = chatWindowEl.scrollHeight;
}

function finalizeAssistantMessage(finalText) {
  if (!currentAssistantBubble) {
     // It's possible a non-streaming message or image came first, or an error cleared it.
     // If finalText is substantial, create a new bubble for it.
     if (finalText && finalText.trim() !== "") {
         console.warn("No current assistant bubble to finalize, but finalText exists. Creating new bubble.");
         const messageObj = { sender: "Nyx", content: finalText };
         appendMessage(messageObj, true);
     } else {
         console.log("No assistant bubble to finalize and no final text.");
     }
     currentAssistantBubble = null; // Ensure reset
     partialAssistantText = "";    // Ensure reset
     return;
  }
  currentAssistantBubble.innerHTML = sanitizeAndRenderMarkdown(finalText);
  currentAssistantBubble = null;
  partialAssistantText = "";
  const chatWindowEl = chatWindow || document.getElementById("chatWindow");
  chatWindowEl.scrollTop = chatWindowEl.scrollHeight;
}

let myUserId = null;

async function checkLoggedIn() {
  const res = await fetch("/whoami", { credentials: "include" });
  const data = await res.json();
  if (!data.logged_in) {
    return window.location.href = "/login_page";
  }
  myUserId = data.user_id;
  document.getElementById("logoutBtn").style.display = "inline-block";
}


function attachEnterKey() {
  (userMsgInput || document.getElementById("userMsg")).addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey) { // Allow shift+enter for new line
      e.preventDefault();
      sendMessage();
    }
  });
}

function loadDarkModeFromStorage() {
  const val = localStorage.getItem("dark_mode_enabled");
  if (val === "true") {
    isDarkMode = true;
    document.body.classList.add("dark-mode");
  } else {
    isDarkMode = false;
    document.body.classList.remove("dark-mode");
  }
}

function toggleDarkMode() {
  isDarkMode = !isDarkMode;
  localStorage.setItem("dark_mode_enabled", isDarkMode);
  document.body.classList.toggle("dark-mode", isDarkMode);
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

async function loadConversations() {
  try {
    const res = await fetch("/multiuser/conversations", { method: "GET", credentials: "include" });
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

async function startNewGame() {
    // Prevent multiple simultaneous game creations
    if (isCreatingGame) {
        console.log("Game creation already in progress");
        return;
    }
    
    // Get the button and disable it
    const newGameBtnEl = newGameBtn || document.getElementById("newGameBtn");
    if (newGameBtnEl) {
        newGameBtnEl.disabled = true;
        newGameBtnEl.textContent = "Creating...";
    }
    
    isCreatingGame = true;
    
    const chatWindowEl = chatWindow || document.getElementById("chatWindow");
    const loadingDiv = document.createElement("div");
    loadingDiv.id = "loadingIndicator";
    loadingDiv.innerHTML = '<div style="text-align: center; padding: 20px; font-style: italic; color: #888;">Initializing new game world...</div>';
    chatWindowEl.appendChild(loadingDiv);
    chatWindowEl.scrollTop = chatWindowEl.scrollHeight;

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

        currentConvId = data.conversation_id;
        messagesOffset = 0;

        // Update loading message
        loadingDiv.innerHTML = '<div style="text-align: center; padding: 20px; font-style: italic; color: #888;">Creating your world... This may take a minute...</div>';

        // Poll for completion
        const maxAttempts = 60; // 60 attempts = 3 minutes max
        let attempts = 0;
        let gameReady = false;

        const pollInterval = setInterval(async () => {
            attempts++;
            
            try {
                const statusRes = await fetch(`/new_game/conversation_status?conversation_id=${currentConvId}`, {
                    method: "GET",
                    credentials: "include"
                });
                
                if (statusRes.ok) {
                    const statusData = await statusRes.json();
                    
                    // Update loading message with status
                    const statusText = attempts % 3 === 0 ? "Still working..." : 
                                     attempts % 3 === 1 ? "Building your world..." : 
                                     "Almost there...";
                    loadingDiv.innerHTML = `<div style="text-align: center; padding: 20px; font-style: italic; color: #888;">${statusText}</div>`;
                    
                    if (statusData.status === "ready") {
                        gameReady = true;
                        clearInterval(pollInterval);
                        
                        // Remove loading indicator
                        if (loadingDiv && loadingDiv.parentNode) {
                            loadingDiv.remove();
                        }
                        
                        // Join socket room
                        if (socket && socket.connected) {
                            socket.emit('join', { conversation_id: currentConvId });
                        }
                        
                        // Load the game content
                        await loadMessages(currentConvId, true);
                        await checkForWelcomeImage(currentConvId);
                        await loadConversations();
                        
                        // Show success message
                        if (statusData.opening_narrative) {
                            // Display the actual opening narrative from Nyx
                            const openingMsg = { 
                                sender: "Nyx", 
                                content: statusData.opening_narrative
                            };
                            appendMessage(openingMsg, true);
                        } else {
                            // Fallback to generic message if no opening narrative
                            const successMsg = { 
                                sender: "system", 
                                content: `New game started! Welcome to ${statusData.conversation_name || "your new world"}.` 
                            };
                            appendMessage(successMsg, true);
                        }
                                                
                    } else if (statusData.status === "failed" || attempts >= maxAttempts) {
                        clearInterval(pollInterval);
                        throw new Error(attempts >= maxAttempts ? "Game creation timed out" : "Game creation failed");
                    }
                }
            } catch (pollError) {
                console.error("Error polling game status:", pollError);
                if (attempts >= maxAttempts) {
                    clearInterval(pollInterval);
                    throw new Error("Failed to check game status");
                }
            }
        }, 3000); // Check every 3 seconds

    } catch (err) {
        console.error("startNewGame error:", err);
        
        // Clean up loading indicator
        const existingLoadingDiv = document.getElementById("loadingIndicator");
        if (existingLoadingDiv) existingLoadingDiv.remove();
        
        // Show error message
        const errorMsg = { 
            sender: "system", 
            content: `Error starting new game: ${err.message}. Please try again.` 
        };
        appendMessage(errorMsg, true);
    } finally {
        // Always re-enable the button and reset the flag
        isCreatingGame = false;
        const newGameBtnEl = newGameBtn || document.getElementById("newGameBtn");
        if (newGameBtnEl) {
            newGameBtnEl.disabled = false;
            newGameBtnEl.textContent = "New Game";
        }
    }
}


function renderConvoList(conversations) {
  const convListDivEl = convListDiv || document.getElementById("convList");
  convListDivEl.innerHTML = "";
  conversations.forEach(conv => {
    const wrapper = document.createElement("div");
    wrapper.style.display = "flex";
    wrapper.style.marginBottom = "5px";
    const btn = document.createElement("button");
    btn.textContent = conv.name || `Game ${conv.id}`;
    btn.style.flex = "1";
    btn.dataset.convId = conv.id; // Store convId for easier access

    // Left-click with protection
    btn.addEventListener("click", async () => {
      if (isSelectingConversation) {
        console.log("Already selecting a conversation");
        return;
      }
      isSelectingConversation = true;
      btn.disabled = true;
      
      try {
        await selectConversation(conv.id);
      } finally {
        isSelectingConversation = false;
        btn.disabled = false;
      }
    });
    
    // Right-click
    btn.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      showContextMenu(e.clientX, e.clientY, conv.id);
    });
    wrapper.appendChild(btn);
    convListDivEl.appendChild(wrapper);
  });
}

async function selectConversation(convId) {
  if (convId === nyxSpaceConvId) {
    console.log("Chat with Nyx selected, clearing window");
  } else {
    console.log(`selectConversation called with ${convId}`);
  }
  currentConvId = convId;
  messagesOffset = 0;
  // Debug socket connection
  console.log(`Socket state: ${socket ? (socket.connected ? 'connected' : 'disconnected') : 'null'}`);
  

  if (socket && socket.connected) {
    socket.emit('join', { conversation_id: convId });
  } else {
    if (socket) socket.connect();
  }

  if (convId === nyxSpaceConvId) {
    (chatWindow || document.getElementById("chatWindow")).innerHTML = ""; // Clear previous
    // Show loading indication while fetching
    appendMessage({sender: "system", content: "Loading Nyx's Space..."}, true);
    // Fetch chat messages
    try {
      let res = await fetch("/nyx_space/messages");
      if (res.ok) {
        let data = await res.json();
        (chatWindow || document.getElementById("chatWindow")).innerHTML = ""; // Remove loading
        if (data.messages && data.messages.length > 0) {
          data.messages.forEach(msg => appendMessage(msg, false)); // false = don't auto-scroll yet
          // After all, then scroll:
          (chatWindow || document.getElementById("chatWindow")).scrollTop = (chatWindow || document.getElementById("chatWindow")).scrollHeight;
        } else {
          appendMessage({sender: "Nyx", content: "Welcome to Nyx's Space! You can chat with me here anytime."}, true);
        }
      } else {
        appendMessage({sender: "Nyx", content: "Could not fetch Nyx's Space messages."}, true);
      }
    } catch (err) {
      (chatWindow || document.getElementById("chatWindow")).innerHTML = "";
      appendMessage({sender: "Nyx", content: "There was an error loading Nyx's Space."}, true);
      console.error(err);
    }
    if (advanceTimeBtn) advanceTimeBtn.style.display = "none";
    if (loadMoreBtn) loadMoreBtn.style.display = "none";
  } else {
    if (advanceTimeBtn) advanceTimeBtn.style.display = "";
    if (loadMoreBtn) loadMoreBtn.style.display = "";
    await loadMessages(convId, true);
    await checkForWelcomeImage(convId);
  }
}


 async function loadMessages(convId, replace = false) {
     const chatWindowEl = chatWindow || document.getElementById("chatWindow");
     const loadMoreBtnEl = loadMoreBtn || document.getElementById("loadMore");
     const url = `/multiuser/conversations/${convId}/messages?offset=${messagesOffset}&limit=${MESSAGES_PER_LOAD}`;
     try {
         const res = await fetch(url, { method: "GET", credentials: "include" });
         if (!res.ok) {
             console.error("Failed to load messages for convo:", convId, res.status);
             appendMessage({sender: "system", content: `Error loading messages for game ${convId}.`}, true);
             return;
         }
         const data = await res.json();
         if (replace) {
             // Clear chat window except for loadMore button
             while (chatWindowEl.firstChild && chatWindowEl.firstChild !== loadMoreBtnEl) {
                 chatWindowEl.removeChild(chatWindowEl.firstChild);
             }
             // If loadMoreBtn was removed, re-add it to the top
             if (!chatWindowEl.contains(loadMoreBtnEl)) {
                 chatWindowEl.insertBefore(loadMoreBtnEl, chatWindowEl.firstChild);
             }
         }
         // Prepend messages if not replacing, append if replacing (after clearing)
         const fragment = document.createDocumentFragment();
         data.messages.slice().reverse().forEach(msg => { // reverse for chronological order when prepending
             fragment.appendChild(createBubble(msg));
         });

         if (replace) {
             chatWindowEl.appendChild(fragment); // Append all new messages
             chatWindowEl.scrollTop = chatWindowEl.scrollHeight;
         } else {
             loadMoreBtnEl.after(fragment); // Insert after "load more"
         }

         loadMoreBtnEl.style.display = data.messages.length < MESSAGES_PER_LOAD ? "none" : "block";
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

function loadPreviousMessages() {
  if (!currentConvId) return;
  messagesOffset += MESSAGES_PER_LOAD;
  loadMessages(currentConvId, false); // false to prepend
}

function appendMessage(m, autoScroll = true) {
  const chatWindowEl = chatWindow || document.getElementById("chatWindow");
  const bubbleRow = createBubble(m);
  chatWindowEl.appendChild(bubbleRow);
  if (autoScroll) {
    chatWindowEl.scrollTop = chatWindowEl.scrollHeight;
  }
  return bubbleRow;
}

function createBubble(m) {
  const row = document.createElement("div");
  row.classList.add("message-row");
  if (m.sender === "user") {
    row.classList.add("user-row");
  } else {
    row.classList.add("gpt-row");
  }
  const bubble = document.createElement("div");
  bubble.classList.add("message-bubble");
  // Sanitize content before setting innerHTML
  const safeContent = sanitizeAndRenderMarkdown(m.content || "");
  bubble.innerHTML = `<strong>${DOMPurify.sanitize(m.sender)}:</strong> ${safeContent}`;
  row.appendChild(bubble);
  return row;
}

async function sendMessage() {
  const userMsgInputEl = userMsgInput || document.getElementById("userMsg");
  const userText = userMsgInputEl.value.trim();
  if (!userText || !currentConvId) return;

  userMsgInputEl.value = "";

  // Nyx's Space handling
  if (currentConvId === nyxSpaceConvId) {
    try {
      // 1. Save the user message
      await fetch('/nyx_space/messages', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({sender: "user", content: userText, timestamp: Date.now()})
      });
  
      appendMessage({sender: "user", content: userText, timestamp: Date.now()}, true);
  
      // 2. Get Nyx's admin reply using the new full admin endpoint!
      // Optional: Expose advanced toggles via UI, or keep it hardcoded for now.
      const adminRequest = {
        user_input: userText,
        // context: { debug_trace: true }, // Optional: add custom admin context
        // use_thinking: true,
        // use_conditioning: false,
        // use_coordination: true,
        // thinking_level: 2,
        // mode: "coordinated",
        // generate_response: true,
        // use_hierarchical_memory: true
      };
      const replyRes = await fetch('/admin/nyx_direct', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(adminRequest)
      });
  
      const replyData = await replyRes.json();
      console.log("Admin Nyx response data:", replyData);
  
      // Safely extract the final message (from response_result or processing_result)
      let aiReply = "...";
      if (replyData.response_result && replyData.response_result.main_message) {
        aiReply = replyData.response_result.main_message;
      } else if (replyData.processing_result && replyData.processing_result.message) {
        aiReply = replyData.processing_result.message;
      } else if (replyData.response_result && replyData.response_result.message) {
        aiReply = replyData.response_result.message;
      }
  
      appendMessage({sender: "Nyx", content: aiReply, timestamp: Date.now()}, true);
  
      // 3. Save Nyx's reply to the DB
      await fetch('/nyx_space/messages', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({sender: "Nyx", content: aiReply, timestamp: Date.now()})
      });
      return;
    } catch (error) {
      console.error("Error processing Nyx admin message:", error);
      appendMessage({sender: "Nyx", content: "Sorry, an error occurred processing your admin message.", timestamp: Date.now()}, true);
      return;
    }
  }

  // Regular conversation handling with Socket.IO
  const userMsgObj = { sender: "user", content: userText };
  appendMessage(userMsgObj, true);

  currentAssistantBubble = null; // Reset for new assistant message
  partialAssistantText = "";

  console.log(`Sending storybeat to server for conversation ${currentConvId}`);
  
  // Check if socket is connected before sending
  if (!socket || !socket.connected) {
    console.warn("Socket disconnected, attempting to reconnect before sending...");
    
    // Add reconnection message
    const reconnectingMsg = { sender: "system", content: "Connection lost. Reconnecting before sending your message..." };
    appendMessage(reconnectingMsg, true);
    
    // Try to reconnect
    if (socket) {
      socket.connect();
      
      // Wait for connection to establish before proceeding
      let attempts = 0;
      const maxAttempts = 5;
      const waitForConnection = setInterval(() => {
        attempts++;
        if (socket.connected) {
          clearInterval(waitForConnection);
          console.log("Socket reconnected. Proceeding with message send.");
          proceedWithSend();
        } else if (attempts >= maxAttempts) {
          clearInterval(waitForConnection);
          console.error("Failed to reconnect after multiple attempts.");
          const errorMsg = { sender: "system", content: "Could not connect to server. Please refresh the page and try again." };
          appendMessage(errorMsg, true);
        }
      }, 1000);
      
      return; // Exit here and let the interval handler call proceedWithSend
    } else {
      const errorMsg = { sender: "system", content: "Connection error. Please refresh the page." };
      appendMessage(errorMsg, true);
      return;
    }
  }
  
  // If socket is connected, proceed with sending
  proceedWithSend();
  
  function proceedWithSend() {
    // Ensure joined to the correct room (socket might have reconnected)
    socket.emit('join', { conversation_id: currentConvId });

    // Set a timeout to detect if server doesn't respond
    const messageTimeout = setTimeout(() => {
      const timeoutMsg = { sender: "system", content: "Server taking longer than expected to respond. Please wait..." };
      appendMessage(timeoutMsg, true);
    }, 15000); // 15 seconds timeout
    
    // Listen for any response to clear the timeout
    const clearTimeoutHandler = () => clearTimeout(messageTimeout);
    socket.once('new_token', clearTimeoutHandler);
    socket.once('error', clearTimeoutHandler);
    socket.once('processing', clearTimeoutHandler);
    
    // Send the message
    socket.emit("storybeat", {
      user_input: userText,
      conversation_id: currentConvId,
      player_name: "Chase", // This should ideally come from logged-in user data
      advance_time: false,
      universal_update: pendingUniversalUpdates
    });
    resetPendingUniversalUpdates();
  }
}

function advanceTime() {
  if (!currentConvId) {
    alert("Please select a conversation first");
    return;
  }
  const userMsg = { sender: "user", content: "Let's advance to the next time period." };
  appendMessage(userMsg, true);

  currentAssistantBubble = null; // Reset
  partialAssistantText = "";

  // Check if socket is connected
  if (!socket || !socket.connected) {
    console.warn("Socket disconnected, attempting to reconnect...");
    
    if (socket) {
      socket.connect();
      
      // Wait briefly for connection
      setTimeout(() => {
        if (socket.connected) {
          sendAdvanceTimeCommand();
        } else {
          const errorMsg = { sender: "system", content: "Not connected to server. Please refresh the page." };
          appendMessage(errorMsg, true);
        }
      }, 1000);
    }
  } else {
    sendAdvanceTimeCommand();
  }
  
  function sendAdvanceTimeCommand() {
    socket.emit('join', { conversation_id: currentConvId }); // Ensure joined
    socket.emit("storybeat", {
      user_input: "Let's advance to the next time period.", // Or a specific system command
      conversation_id: currentConvId,
      player_name: "Chase",
      advance_time: true,
      universal_update: pendingUniversalUpdates
    });
    resetPendingUniversalUpdates();
  }
}

// Context Menu
function showContextMenu(x, y, convId) {
  const menuDivEl = contextMenuDiv || document.getElementById("contextMenu");
  menuDivEl.innerHTML = ""; // Clear previous items

  const options = [
    { text: "Rename", action: () => renameConversation(convId) },
    { text: "Move to Folder", action: () => moveConversationToFolder(convId) },
    { text: "Delete", action: () => deleteConversation(convId) }
  ];

  options.forEach(opt => {
    const div = document.createElement("div");
    div.textContent = opt.text;
    div.addEventListener("click", (e) => {
      e.stopPropagation(); // Prevent click from closing menu immediately via document listener
      opt.action();
      menuDivEl.style.display = "none"; // Hide menu after action
    });
    menuDivEl.appendChild(div);
  });

  menuDivEl.style.left = x + "px";
  menuDivEl.style.top = y + "px";
  menuDivEl.style.display = "block";
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
      await loadConversations(); // Refresh list
    } else {
      const errData = await res.json().catch(() => ({error: "Unknown error during rename"}));
      alert(`Error renaming conversation: ${errData.error}`);
    }
  } catch (err) {
    console.error("Rename conversation fetch error:", err);
    alert("Network error renaming conversation.");
  }
}

async function moveConversationToFolder(convId) {
  const folderName = prompt("Enter folder name (will be created if it doesn't exist):");
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
     const errData = await res.json().catch(() => ({error: "Unknown error moving conversation"}));
      alert(`Error moving conversation: ${errData.error}`);
    }
  } catch (err) {
    console.error("Move conversation fetch error:", err);
    alert("Network error moving conversation.");
  }
}

async function deleteConversation(convId) {
  if (!confirm("Are you sure you want to delete this conversation? This cannot be undone.")) return;
  try {
    const res = await fetch(`/multiuser/conversations/${convId}`, {
      method: "DELETE",
      credentials: "include"
    });
    if (res.ok) {
      await loadConversations();
      if (currentConvId === convId) {
        (chatWindow || document.getElementById("chatWindow")).innerHTML = '<span id="loadMore" style="display:none;">Load older messages...</span>'; // Clear chat if current was deleted
        currentConvId = null;
      }
    } else {
     const errData = await res.json().catch(() => ({error: "Unknown error deleting conversation"}));
      alert(`Error deleting conversation: ${errData.error}`);
    }
  } catch (err) {
    console.error("Delete conversation fetch error:", err);
    alert("Network error deleting conversation.");
  }
}

// Improved Socket.IO initialization with robust configuration
function initializeSocket() {
  // Use the enhanced connection manager
  socket = createRobustSocketConnection({
    // Event handlers that integrate with the UI
    onConnect: (socket, wasReconnect) => {
      console.log("Socket connected with ID:", socket.id);
      
      // If reconnection was in progress, add a system message
      if (wasReconnect) {
        const reconnectMsg = { sender: "system", content: "Connection restored!" };
        appendMessage(reconnectMsg, true);
        reconnectionInProgress = false;
      }
      
      // Join conversation room if one is active
      if (currentConvId) {
        socket.emit('join', { conversation_id: currentConvId });
        console.log(`Joined room: ${currentConvId}`);
      }
    },
    
    onDisconnect: (socket, reason) => {
      console.error("Socket disconnected:", reason);
      reconnectionInProgress = true;
      const disconnectMsg = { sender: "system", content: `Connection lost (${reason}). Attempting to reconnect...` };
      appendMessage(disconnectMsg, true);
    },
    
    onReconnect: (socket, attemptNumber) => {
      console.log(`Socket reconnected after ${attemptNumber} attempts`);
      reconnectionInProgress = false;
      
      // Rejoin the conversation room if active
      if (currentConvId) {
        socket.emit('join', { conversation_id: currentConvId });
      }
      
      const reconnectMsg = { sender: "system", content: "Connection restored!" };
      appendMessage(reconnectMsg, true);
    },
    
    onReconnectFailed: () => {
      console.error("Socket reconnection failed");
      const failMsg = { sender: "system", content: "Unable to reconnect. Please refresh the page." };
      appendMessage(failMsg, true);
    }
  });
  
  // Set up event listeners for message handling
  setupSocketMessageListeners();
}

// Separate function for message-specific listeners
function setupSocketMessageListeners() {
  // Room events
  socket.on("joined", (data) => {
    console.log("Joined room:", data.room);
    const joinMsg = { sender: "system", content: "Connected to game session" };
    appendMessage(joinMsg, true);
  });

  // Message streaming events
  socket.on("new_token", function(payload) {
    handleNewToken(payload.token);
  });

  socket.on("done", function(payload) {
    console.log("Done streaming. Full text received.");
    finalizeAssistantMessage(payload.full_text);
  });

  socket.on("error", (payload) => {
    console.error("Server error:", payload.error);
    handleNewToken("[Error: " + payload.error + "]");
    finalizeAssistantMessage(""); // Finalize with empty if error occurred during stream
  });

  socket.on("message", function(data) {
    console.log("Received non-streaming message event:", data);
    const messageObj = {
      sender: data.sender || "Nyx",
      content: data.content || data.message || "No content"
    };
    appendMessage(messageObj, true);
  });

  socket.on("image", function(payload) {
    console.log("Received image:", payload);
    appendImageToChat(payload.image_url, payload.reason);
  });

  socket.on("processing", function(data) {
    console.log("Server is processing:", data.message);
    const chatWindowEl = document.getElementById("chatWindow");
    let processingDiv = document.getElementById("processingIndicator");
    if (!processingDiv) {
        processingDiv = document.createElement("div");
        processingDiv.id = "processingIndicator";
        processingDiv.innerHTML = '<div style="text-align: center; padding: 10px; font-style: italic; color: #888;">Processing your request...</div>';
        chatWindowEl.appendChild(processingDiv);
    }
    chatWindowEl.scrollTop = chatWindowEl.scrollHeight;
  });

  socket.on("game_state_update", function(data) {
    console.log("Game state updated:", data);
    // Example: if (data.type === "npc_update") updateNPCInfo(data.npc_data);
  });
}

// DOMContentLoaded to initialize everything
const nyxSpaceConvId = "__nyx_space__";

document.addEventListener('DOMContentLoaded', async function() {
  // Cache DOM elements
  logoutBtn = document.getElementById("logoutBtn");
  toggleDarkModeBtn = document.getElementById("toggleDarkModeBtn");
  advanceTimeBtn = document.getElementById("advanceTimeBtn");
  newGameBtn = document.getElementById("newGameBtn");
  const nyxSpaceBtn = document.getElementById("nyxSpaceBtn");
  if (nyxSpaceBtn) nyxSpaceBtn.addEventListener("click", () => selectConversation(nyxSpaceConvId));
  convListDiv = document.getElementById("convList");
  chatWindow = document.getElementById("chatWindow");
  loadMoreBtn = document.getElementById("loadMore");
  userMsgInput = document.getElementById("userMsg");
  sendBtn = document.getElementById("sendBtn");
  contextMenuDiv = document.getElementById("contextMenu");
  leftPanelInner = document.getElementById("leftPanelInner");

  // Initial setup
  await checkLoggedIn(); // This might redirect, so subsequent calls might not happen if not logged in
  attachEnterKey();
  loadDarkModeFromStorage();
  await loadConversations();

  // Socket.IO connection with improved configuration
  initializeSocket();

  // Attach global event listeners
  if (logoutBtn) logoutBtn.addEventListener("click", logout);
  if (toggleDarkModeBtn) toggleDarkModeBtn.addEventListener("click", toggleDarkMode);
  if (advanceTimeBtn) advanceTimeBtn.addEventListener("click", advanceTime);
  if (newGameBtn) newGameBtn.addEventListener("click", startNewGame);
  if (loadMoreBtn) loadMoreBtn.addEventListener("click", loadPreviousMessages);
  if (sendBtn) sendBtn.addEventListener("click", sendMessage);

  // Hide context menu on global click
  document.addEventListener("click", (e) => {
     if (contextMenuDiv && !contextMenuDiv.contains(e.target)) {
         contextMenuDiv.style.display = "none";
     }
  });
  
  // Add page visibility change handler to detect when browser tab becomes inactive/active
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      console.log("Page is now visible, checking connection status");
      // If socket exists but disconnected, try to reconnect
      if (socket && !socket.connected) {
        console.log("Page became visible but socket disconnected. Reconnecting...");
        socket.connect();
      }
    }
  });
  
  // Add window focus handler as well (sometimes more reliable than visibilitychange)
  window.addEventListener("focus", () => {
    console.log("Window regained focus, checking connection status");
    if (socket && !socket.connected) {
      console.log("Window focused but socket disconnected. Reconnecting...");
      socket.connect();
    }
  });
  
  // Add window online/offline handlers
  window.addEventListener("online", () => {
    console.log("Browser reports online status, attempting to reconnect socket");
    if (socket && !socket.connected) {
      socket.connect();
    }
  });
  
  window.addEventListener("offline", () => {
    console.log("Browser reports offline status");
    // No action needed - the socket's internal events will handle this
  });
});
