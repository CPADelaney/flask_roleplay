// socket-fix.js
(function() {
  // Configuration and state
  const DEBUG = true;
  let lastSocketId = null;
  let reconnectAttempts = 0;
  let isConnecting = false;
  let healthCheckInterval = null;
  let heartbeatInterval = null;
  
  // Debug logging function
  function debugLog(...args) {
    if (DEBUG && console) {
      console.log(`[SocketFix]`, ...args);
    }
  }
  
  /**
   * Creates a robust Socket.IO connection with enhanced reliability
   * @param {Object} options - Configuration options
   * @returns {SocketIO.Socket} - The configured socket instance
   */
  window.createRobustSocketConnection = function(options = {}) {
    if (typeof io === 'undefined') {
      console.error('Socket.IO not available');
      return null;
    }
    
    // Clean up any existing resources
    cleanupExistingResources();
    
    // Combine default options with provided options
    const config = {
      path: '/socket.io',
      transports: ['websocket', 'polling'],
      auth: { user_id: window.CURRENT_USER_ID || 'anonymous' },
      reconnection: true,
      reconnectionAttempts: 10, // Limit max attempts to prevent flooding
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 60000, // Reduce from 120000
      pingTimeout: 60000, // Reduce from 120000
      pingInterval: 25000,
      forceNew: false,
      ...options
    };
    
    debugLog('Creating socket connection with config:', config);
    
    // Create the Socket.IO connection
    const socket = io(config);
    
    // Set up enhanced event handlers
    setupSocketEventHandlers(socket, options);
    
    // Set up browser event handlers for added reliability
    setupBrowserEventHandlers(socket);
    
    // Start health monitoring
    startHealthMonitoring(socket);
    
    return socket;
  };
  
  /**
   * Set up Socket.IO event handlers with better error handling
   */
  function setupSocketEventHandlers(socket, userOptions) {
    socket.on('connect', () => {
      const wasReconnect = lastSocketId !== null && lastSocketId !== socket.id;
      lastSocketId = socket.id;
      isConnecting = false;
      reconnectAttempts = 0;
      
      debugLog(`Socket connected! ID: ${socket.id}, Was reconnect: ${wasReconnect}`);
      startHeartbeat(socket);
      
      // Call user-defined handler if provided
      if (typeof userOptions.onConnect === 'function') {
        userOptions.onConnect(socket, wasReconnect);
      }
    });
    
    socket.on('disconnect', (reason) => {
      debugLog(`Socket disconnected: ${reason}`);
      isConnecting = false;
      stopHeartbeat();
      
      if (typeof userOptions.onDisconnect === 'function') {
        userOptions.onDisconnect(socket, reason);
      }
      
      // Handle server-initiated disconnects
      if (reason === 'io server disconnect') {
        debugLog('Server initiated disconnect, manually reconnecting after delay...');
        setTimeout(() => {
          if (!socket.connected) socket.connect();
        }, 2000);
      }
      
      // Handle transport errors by forcing a new connection
      if (reason === 'transport error' || reason === 'transport close') {
        debugLog('Transport error/close, attempting recovery...');
        if (reconnectAttempts > 3 && !isConnecting) {
          isConnecting = true;
          debugLog('Multiple reconnect failures, forcing new connection...');
          setTimeout(() => {
            socket.close();
            socket.connect();
          }, 2000);
        }
      }
    });
    
    socket.on('reconnect_attempt', (attemptNumber) => {
      reconnectAttempts = attemptNumber;
      debugLog(`Reconnection attempt #${attemptNumber}`);
      
      if (typeof userOptions.onReconnectAttempt === 'function') {
        userOptions.onReconnectAttempt(socket, attemptNumber);
      }
      
      // After multiple failures, try alternate transport strategy
      if (attemptNumber % 3 === 0) {
        debugLog('Multiple reconnect attempts, trying alternate transport');
        socket.io.opts.transports = ['polling', 'websocket'];
      }
    });
    
    socket.on('reconnect', (attemptNumber) => {
      debugLog(`Reconnected after ${attemptNumber} attempts!`);
      reconnectAttempts = 0;
      
      if (typeof userOptions.onReconnect === 'function') {
        userOptions.onReconnect(socket, attemptNumber);
      }
    });
    
    socket.on('reconnect_failed', () => {
      debugLog('Reconnection failed after all attempts');
      
      if (typeof userOptions.onReconnectFailed === 'function') {
        userOptions.onReconnectFailed(socket);
      }
    });
    
    socket.on('error', (error) => {
      debugLog('Socket error:', error);
      
      if (typeof userOptions.onError === 'function') {
        userOptions.onError(socket, error);
      }
    });
    
    socket.on('connect_error', (error) => {
      debugLog('Connection error:', error);
      
      if (typeof userOptions.onConnectError === 'function') {
        userOptions.onConnectError(socket, error);
      }
    });
    
    // Server heartbeat response handler
    socket.on('server_heartbeat', (data) => {
      debugLog('Received server heartbeat:', data.timestamp);
    });
  }
  
  /**
   * Set up browser event handlers to improve connection reliability
   */
  function setupBrowserEventHandlers(socket) {
    // Visibility change handler (tab focus/background)
    document.addEventListener('visibilitychange', () => {
      debugLog(`Visibility changed: ${document.visibilityState}`);
      
      if (document.visibilityState === 'visible') {
        debugLog('Page became visible, checking connection');
        if (!socket.connected && !isConnecting) {
          debugLog('Socket disconnected while page was hidden, reconnecting');
          isConnecting = true;
          socket.connect();
        }
      }
    });
    
    // Window focus handler for additional reliability
    window.addEventListener('focus', () => {
      debugLog('Window focused, checking connection');
      if (!socket.connected && !isConnecting) {
        debugLog('Socket disconnected when window was unfocused, reconnecting');
        isConnecting = true;
        socket.connect();
      }
    });
    
    // Online/offline handlers
    window.addEventListener('online', () => {
      debugLog('Browser reports online status, attempting to reconnect');
      if (!socket.connected && !isConnecting) {
        isConnecting = true;
        socket.connect();
      }
    });
    
    window.addEventListener('offline', () => {
      debugLog('Browser reports offline status');
      // The socket's internal handlers will manage this
    });
    
    // Before unload handler to clean up
    window.addEventListener('beforeunload', () => {
      debugLog('Page unloading, cleaning up socket');
      cleanupExistingResources();
    });
  }
  
  /**
   * Start regular health check monitoring
   */
  function startHealthMonitoring(socket) {
    stopHealthMonitoring(); // Clear any existing interval
    
    healthCheckInterval = setInterval(() => {
      if (socket && !socket.connected && !isConnecting) {
        debugLog('Health check: Socket disconnected but not reconnecting. Forcing reconnection...');
        isConnecting = true;
        socket.connect();
        
        // If still not connected after a delay, force a new connection
        setTimeout(() => {
          if (!socket.connected) {
            debugLog('Still not connected after reconnect attempt, forcing close and reconnect');
            socket.close();
            socket.connect();
          }
          isConnecting = false;
        }, 3000);
      }
    }, 15000); // Check every 15 seconds
  }
  
  /**
   * Stop health monitoring
   */
  function stopHealthMonitoring() {
    if (healthCheckInterval) {
      clearInterval(healthCheckInterval);
      healthCheckInterval = null;
    }
  }
  
  /**
   * Start regular heartbeat to keep the connection alive
   */
  function startHeartbeat(socket) {
    stopHeartbeat(); // Clear any existing interval
    
    heartbeatInterval = setInterval(() => {
      if (socket && socket.connected) {
        socket.emit('client_heartbeat', { timestamp: Date.now() });
        debugLog('Sent heartbeat ping');
      } else {
        debugLog('Skipped heartbeat - socket not connected');
      }
    }, 20000); // Every 20 seconds
  }
  
  /**
   * Stop heartbeat
   */
  function stopHeartbeat() {
    if (heartbeatInterval) {
      clearInterval(heartbeatInterval);
      heartbeatInterval = null;
    }
  }
  
  /**
   * Clean up any existing resources
   */
  function cleanupExistingResources() {
    stopHealthMonitoring();
    stopHeartbeat();
  }
})();
