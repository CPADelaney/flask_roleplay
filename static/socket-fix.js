// static/socket-fix.js
(function() {
  // Store last connected socket ID to detect reconnections with new IDs
  let lastSocketId = null;
  let reconnectAttempts = 0;
  let isConnecting = false;
  
  // Detect if this is a browser environment
  const isBrowser = typeof window !== 'undefined';
  
  // Debug mode - enable for detailed logging
  const DEBUG = true;
  
  // Helper logging function
  function debugLog(...args) {
    if (DEBUG && console) {
      console.log(`[SocketFix]`, ...args);
    }
  }

  function createRobustSocketConnection() {
    const socket = io({
      path: '/socket.io',
      transports: ['websocket', 'polling'], // Try websocket first, fallback to polling
      auth: { user_id: window.CURRENT_USER_ID },
      reconnection: true,
      reconnectionAttempts: 10, // Limit max attempts to prevent flooding
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 60000, // Reduce from 120000 to match more common configurations
      pingTimeout: 60000, // Reduce from 120000
      pingInterval: 25000
    });
    
    return socket;
  }
  

    debugLog('Creating new socket connection with config:', config);
    
    // Create the Socket.IO connection
    const socket = io(config);
    
    // Set up enhanced connection event handlers
    socket.on('connect', () => {
      const wasReconnect = lastSocketId !== null && lastSocketId !== socket.id;
      lastSocketId = socket.id;
      isConnecting = false;
      reconnectAttempts = 0;
      
      debugLog(`Socket connected! ID: ${socket.id}, Was reconnect: ${wasReconnect}`);
      
      // Call any user-defined connect handler
      if (typeof options.onConnect === 'function') {
        options.onConnect(socket, wasReconnect);
      }
      
      // Ping socket periodically to keep connection alive
      startKeepalive(socket);
    });
    
    // Enhanced disconnect handling
    socket.on('disconnect', (reason) => {
      debugLog(`Socket disconnected: ${reason}`);
      isConnecting = false;
      
      // Stop keepalive to avoid timer leaks
      stopKeepalive();
      
      if (typeof options.onDisconnect === 'function') {
        options.onDisconnect(socket, reason);
      }
      
      // Handle server-initiated disconnects
      if (reason === 'io server disconnect') {
        debugLog('Server initiated disconnect, manually reconnecting...');
        // Wait a moment to prevent immediate reconnection storms
        setTimeout(() => socket.connect(), 1000);
      }
      
      // Handle transport errors by forcing a new connection
      if (reason === 'transport error' || reason === 'transport close') {
        debugLog('Transport error/close, attempting recovery...');
        // If many consecutive errors, try forcing a new connection
        if (reconnectAttempts > 3 && !isConnecting) {
          isConnecting = true;
          debugLog('Multiple reconnect failures, forcing new connection...');
          // Force a complete new connection after multiple failures
          setTimeout(() => {
            socket.close();
            socket.connect();
          }, 2000);
        }
      }
    });
    
    // Track reconnection attempts
    socket.on('reconnect_attempt', (attemptNumber) => {
      reconnectAttempts = attemptNumber;
      debugLog(`Reconnection attempt #${attemptNumber}`);
      
      if (typeof options.onReconnectAttempt === 'function') {
        options.onReconnectAttempt(socket, attemptNumber);
      }
      
      // After multiple failed attempts, try switching transports
      if (attemptNumber % 3 === 0) {
        debugLog('Multiple reconnect attempts, trying alternate transport');
        socket.io.opts.transports = ['polling', 'websocket'];
      }
    });
    
    // Successful reconnection
    socket.on('reconnect', (attemptNumber) => {
      debugLog(`Reconnected after ${attemptNumber} attempts!`);
      reconnectAttempts = 0;
      
      if (typeof options.onReconnect === 'function') {
        options.onReconnect(socket, attemptNumber);
      }
    });
    
    // Failed reconnection after all attempts
    socket.on('reconnect_failed', () => {
      debugLog('Reconnection failed after all attempts');
      
      if (typeof options.onReconnectFailed === 'function') {
        options.onReconnectFailed(socket);
      }
    });
    
    // Error handling
    socket.on('error', (error) => {
      debugLog('Socket error:', error);
      
      if (typeof options.onError === 'function') {
        options.onError(socket, error);
      }
    });
    
    // Connection error handling
    socket.on('connect_error', (error) => {
      debugLog('Connection error:', error);
      
      if (typeof options.onConnectError === 'function') {
        options.onConnectError(socket, error);
      }
    });
    
    // --- Keepalive mechanism ---
    let keepaliveTimer = null;
    
    function startKeepalive(socket) {
      // Clear any existing timer
      stopKeepalive();
      
      // Create new keepalive timer
      keepaliveTimer = setInterval(() => {
        if (socket.connected) {
          // Send a custom heartbeat event
          socket.emit('client_heartbeat', { timestamp: Date.now() });
          debugLog('Sent keepalive heartbeat');
        } else {
          debugLog('Skipping keepalive - socket not connected');
        }
      }, 20000); // Every 20 seconds - less than the server's pingInterval
    }
    
    function stopKeepalive() {
      if (keepaliveTimer) {
        clearInterval(keepaliveTimer);
        keepaliveTimer = null;
      }
    }
    
    // Add method to safely clean up the socket
    socket.cleanup = function() {
      debugLog('Cleaning up socket connection');
      stopKeepalive();
      this.removeAllListeners();
      this.close();
    };
    
    return socket;
  };
})();
