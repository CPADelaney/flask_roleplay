// login_page.js

// Function to check if user is already logged in
async function checkAlreadyLoggedIn() {
  try {
    const res = await fetch("/whoami", {
      method: "GET",
      credentials: "include" // Send cookies
    });
    if (res.ok) {
      const data = await res.json();
      if (data.logged_in) {
        // If they're logged in, redirect to the chat page
        window.location.href = "/chat";
      }
    }
  } catch (err) {
    console.log("No login session found or an error occurred during checkAlreadyLoggedIn:", err);
    // Don't redirect, stay on login page
  }
}

// Function for user registration
async function registerUser() {
  const user = document.getElementById("reg_username").value.trim();
  const pass = document.getElementById("reg_password").value.trim();
  const statusDiv = document.getElementById("registerStatus");

  console.log("Registering user:", user); // For debugging, avoid logging pass in production

  if (!user || !pass) {
    statusDiv.innerText = "Username & password required to register.";
    statusDiv.style.color = "red";
    return;
  }
  statusDiv.innerText = "Registering...";
  statusDiv.style.color = "black";

  try {
    const res = await fetch("/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include", // Send/receive session cookies
      body: JSON.stringify({ username: user, password: pass })
    });
    const data = await res.json();
    if (res.ok) {
      statusDiv.innerText = `Registration successful! Welcome, ${data.username || user}! Redirecting...`;
      statusDiv.style.color = "green";
      // Redirect to chat page on successful registration
      window.location.href = "/chat";
    } else {
      statusDiv.innerText = `Registration error: ${data.error || "Unknown error"}`;
      statusDiv.style.color = "red";
    }
  } catch (err) {
    statusDiv.innerText = "Registration request failed. Please try again or check the console.";
    statusDiv.style.color = "red";
    console.error("Registration error:", err);
  }
}

// Function for user login
async function attemptLogin() {
  const user = document.getElementById("username").value.trim();
  const pass = document.getElementById("password").value.trim();
  const statusDiv = document.getElementById("loginStatus");

  console.log("Attempting login with:", user); // For debugging, avoid logging pass in production

  if (!user || !pass) {
    statusDiv.innerText = "Username & password required to login.";
    statusDiv.style.color = "red";
    return;
  }
  statusDiv.innerText = "Logging in...";
  statusDiv.style.color = "black";

  try {
    const res = await fetch("/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include", // Send/receive session cookies
      body: JSON.stringify({ username: user, password: pass })
    });
    const data = await res.json();
    if (res.ok) {
      statusDiv.innerText = `Login successful! Welcome back, ${data.username || user}! Redirecting...`;
      statusDiv.style.color = "green";
      // Redirect to chat page on successful login
      window.location.href = "/chat";
    } else {
      statusDiv.innerText = `Login error: ${data.error || "Unknown error"}`;
      statusDiv.style.color = "red";
    }
  } catch (err) {
    statusDiv.innerText = "Login request failed. Please try again or check the console.";
    statusDiv.style.color = "red";
    console.error("Login error:", err);
  }
}

// Wait for the DOM to be fully loaded before running scripts that interact with it
document.addEventListener('DOMContentLoaded', async function() {
  // Check if user is already logged in when the page loads
  await checkAlreadyLoggedIn();

  // Attach event listener to the Register button
  const registerButton = document.getElementById("registerBtn");
  if (registerButton) {
    registerButton.addEventListener("click", registerUser);
  }

  // Attach event listener to the Login button
  const loginButton = document.getElementById("loginBtn");
  if (loginButton) {
    loginButton.addEventListener("click", attemptLogin);
  }

  // Optional: Add Enter key press listeners for input fields
  document.getElementById("reg_password").addEventListener("keypress", function(event) {
     if (event.key === "Enter") {
         event.preventDefault(); // Prevent default form submission if it were a form
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
