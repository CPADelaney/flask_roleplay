// static/login_page.js

// Function to check if user is already logged in
async function checkAlreadyLoggedIn() {
  try {
    const res = await fetch("/whoami", {
      method: "GET",
      credentials: "include"
    });
    if (res.ok) {
      const data = await res.json();
      if (data.logged_in) {
        window.location.href = "/chat";
      }
    }
  } catch (err) {
    console.log("No login session found or an error occurred during checkAlreadyLoggedIn:", err);
  }
}

// Helper function to set status message and class
function setStatus(statusDiv, message, type) {
  statusDiv.innerText = message;
  // Remove previous status types and add the new one, keeping 'status-message'
  statusDiv.className = 'status-message'; // Reset to base class
  if (type) {
    statusDiv.classList.add(`status-${type}`);
  }
}

// Function for user registration
async function registerUser() {
  const user = document.getElementById("reg_username").value.trim();
  const pass = document.getElementById("reg_password").value.trim();
  const statusDiv = document.getElementById("registerStatus");

  console.log("Registering user:", user);

  if (!user || !pass) {
    setStatus(statusDiv, "Username & password required to register.", "error");
    return;
  }
  setStatus(statusDiv, "Registering...", "processing");

  try {
    const res = await fetch("/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ username: user, password: pass })
    });
    const data = await res.json();
    if (res.ok) {
      setStatus(statusDiv, `Registration successful! Welcome, ${data.username || user}! Redirecting...`, "success");
      setTimeout(() => { window.location.href = "/chat"; }, 1500); // Give user time to see message
    } else {
      setStatus(statusDiv, `Registration error: ${data.error || "Unknown error"}`, "error");
    }
  } catch (err) {
    setStatus(statusDiv, "Registration request failed. Please try again or check the console.", "error");
    console.error("Registration error:", err);
  }
}

// Function for user login
async function attemptLogin() {
  const user = document.getElementById("username").value.trim();
  const pass = document.getElementById("password").value.trim();
  const statusDiv = document.getElementById("loginStatus");

  console.log("Attempting login with:", user);

  if (!user || !pass) {
    setStatus(statusDiv, "Username & password required to login.", "error");
    return;
  }
  setStatus(statusDiv, "Logging in...", "processing");

  try {
    const res = await fetch("/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ username: user, password: pass })
    });
    const data = await res.json();
    if (res.ok) {
      setStatus(statusDiv, `Login successful! Welcome back, ${data.username || user}! Redirecting...`, "success");
      setTimeout(() => { window.location.href = "/chat"; }, 1500); // Give user time to see message
    } else {
      setStatus(statusDiv, `Login error: ${data.error || "Unknown error"}`, "error");
    }
  } catch (err) {
    setStatus(statusDiv, "Login request failed. Please try again or check the console.", "error");
    console.error("Login error:", err);
  }
}

document.addEventListener('DOMContentLoaded', async function() {
  await checkAlreadyLoggedIn();

  const registerButton = document.getElementById("registerBtn");
  if (registerButton) {
    registerButton.addEventListener("click", registerUser);
  }

  const loginButton = document.getElementById("loginBtn");
  if (loginButton) {
    loginButton.addEventListener("click", attemptLogin);
  }

  document.getElementById("reg_password").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      registerUser();
    }
  });
  document.getElementById("password").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      attemptLogin();
    }
  });
});
