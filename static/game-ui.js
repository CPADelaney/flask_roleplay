// game-ui.js - Game UI functionality for tabs and data display

// Initialize admin UI
function initializeAdminUI() {
  if (window.IS_ADMIN) {
    document.body.classList.add('is-admin');
  }
}

// Tab Management
function initializeTabs() {
  const tabs = document.querySelectorAll('.tab');
  const tabPanes = document.querySelectorAll('.tab-pane');
  
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      // Remove active class from all tabs and panes
      tabs.forEach(t => t.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));
      
      // Add active class to clicked tab
      tab.classList.add('active');
      
      // Show corresponding pane
      const tabName = tab.dataset.tab;
      const pane = document.getElementById(tabName + 'Tab');
      if (pane) {
        pane.classList.add('active');
        
        // Load data for the active tab
        loadTabData(tabName);
      }
    });
  });
}

// Load data based on active tab
async function loadTabData(tabName) {
  if (!AppState.currentConvId) return;
  
  switch(tabName) {
    case 'stats':
      await loadPlayerStats();
      break;
    case 'calendar':
      await loadCalendar();
      break;
    case 'journal':
      await loadJournal();
      break;
    case 'inventory':
      await loadInventory();
      break;
    case 'social':
      await loadSocialLinks();
      break;
    case 'codex':
      await loadCodex();
      break;
    case 'settings':
      // Settings tab doesn't need data loading yet
      break;
  }
}

// Dynamic Stats Loading
async function loadPlayerStats() {
  try {
    // Get aggregated roleplay context which includes player stats
    const data = await fetchJson(`/universal/get_aggregated_roleplay_context?conversation_id=${AppState.currentConvId}&player_name=Chase`);
    
    // Display player stats dynamically based on what's available
    const statsContainer = document.getElementById('playerStats');
    if (data.playerStats) {
      const stats = data.playerStats;
      let statsHTML = '';
      
      // Define all possible stats with their display properties
      const statDefinitions = {
        corruption: { label: 'Corruption', color: 'default' },
        confidence: { label: 'Confidence', color: 'default' },
        willpower: { label: 'Willpower', color: 'default' },
        obedience: { label: 'Obedience', color: 'default' },
        dependency: { label: 'Dependency', color: 'default' },
        lust: { label: 'Lust', color: 'default' },
        intelligence: { label: 'Intelligence', color: 'default' },
        strength: { label: 'Strength', color: 'default' },
        charisma: { label: 'Charisma', color: 'default' },
        dexterity: { label: 'Dexterity', color: 'default' },
        wisdom: { label: 'Wisdom', color: 'default' },
        constitution: { label: 'Constitution', color: 'default' }
      };
      
      // Dynamically add stats that exist
      for (const [statKey, statDef] of Object.entries(statDefinitions)) {
        if (stats.hasOwnProperty(statKey) && stats[statKey] !== null && stats[statKey] !== undefined) {
          statsHTML += `
            <div class="stat-item">
              <div class="stat-label">${statDef.label}</div>
              <div class="stat-value">${stats[statKey]}</div>
              <div class="progress-bar">
                <div class="progress-fill" style="width: ${Math.min(100, Math.max(0, stats[statKey]))}%"></div>
              </div>
            </div>
          `;
        }
      }
      
      // Also check for any custom stats not in our definitions
      for (const [key, value] of Object.entries(stats)) {
        if (!statDefinitions.hasOwnProperty(key) && value !== null && value !== undefined) {
          // Format the key nicely (snake_case to Title Case)
          const label = key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
          statsHTML += `
            <div class="stat-item">
              <div class="stat-label">${label}</div>
              <div class="stat-value">${value}</div>
              ${typeof value === 'number' && value >= 0 && value <= 100 ? `
                <div class="progress-bar">
                  <div class="progress-fill" style="width: ${value}%"></div>
                </div>
              ` : ''}
            </div>
          `;
        }
      }
      
      statsContainer.innerHTML = statsHTML || '<p>No stats available yet</p>';
    }
    
    // Get and display resources
    const resourceData = await fetchJson(`/universal/player/resources?conversation_id=${AppState.currentConvId}&player_name=Chase`);

    
    const resourceContainer = document.getElementById('playerResources');
    if (resourceData.resources) {
      const res = resourceData.resources;
      let resourceHTML = '';
      
      // Dynamically display all available resources
      for (const [key, value] of Object.entries(res)) {
        if (value !== null && value !== undefined) {
          const label = key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
          const displayValue = typeof value === 'number' && key === 'money' ? `$${value}` : value;
          resourceHTML += `
            <div class="stat-item">
              <div class="stat-label">${label}</div>
              <div class="stat-value">${displayValue}</div>
            </div>
          `;
        }
      }
      
      resourceContainer.innerHTML = resourceHTML || '<p>No resources tracked yet</p>';
    }
    
    const vitalsContainer = document.getElementById('playerVitals');
    if (resourceData.vitals) {
      const vitals = resourceData.vitals;
      let vitalsHTML = '';
      
      const vitalDefinitions = {
        hunger: { label: 'Hunger', gradient: 'linear-gradient(to right, #ff9500, #ff3b30)' },
        energy: { label: 'Energy', gradient: 'linear-gradient(to right, #34c759, #30d158)' },
        health: { label: 'Health', gradient: 'linear-gradient(to right, #ff3b30, #ff2d55)' },
        stamina: { label: 'Stamina', gradient: 'linear-gradient(to right, #5ac8fa, #007aff)' },
        sanity: { label: 'Sanity', gradient: 'linear-gradient(to right, #af52de, #5e5ce6)' }
      };
      
      for (const [vitalKey, vitalDef] of Object.entries(vitalDefinitions)) {
        if (vitals.hasOwnProperty(vitalKey) && vitals[vitalKey] !== null && vitals[vitalKey] !== undefined) {
          vitalsHTML += `
            <div class="stat-item">
              <div class="stat-label">${vitalDef.label}</div>
              <div class="stat-value">${vitals[vitalKey]}</div>
              <div class="progress-bar">
                <div class="progress-fill" style="width: ${Math.min(100, Math.max(0, vitals[vitalKey]))}%; background: ${vitalDef.gradient};"></div>
              </div>
            </div>
          `;
        }
      }
      
      vitalsContainer.innerHTML = vitalsHTML || '<p>No vitals tracked yet</p>';
    }
  } catch (error) {
    console.error('Error loading player stats:', error);
  }
}

// Calendar
async function loadCalendar() {
  try {
    const data = await fetchJson(`/universal/get_aggregated_roleplay_context?conversation_id=${AppState.currentConvId}&player_name=Chase`);
    
    // Display current date
    const currentDateEl = document.getElementById('currentDate');
    const calendarTitle = document.getElementById('calendarTitle');
    
    if (data.currentRoleplay) {
      const year = data.year || 1;
      const month = data.month || 1;
      const day = data.day || 1;
      const timeOfDay = data.timeOfDay || 'Morning';
      
      // Get calendar names if available
      const calendarNames = data.currentRoleplay.CalendarNames;
      let yearName = `Year ${year}`;
      let monthName = `Month ${month}`;
      
      if (calendarNames) {
        try {
          const names = typeof calendarNames === 'string' ? JSON.parse(calendarNames) : calendarNames;
          yearName = names.year_name || yearName;
          if (names.months && names.months[month - 1]) {
            monthName = names.months[month - 1];
          }
        } catch (e) {
          console.error('Error parsing calendar names:', e);
        }
      }
      
      currentDateEl.textContent = `${monthName} ${day}, ${yearName} - ${timeOfDay}`;
      calendarTitle.textContent = yearName;
      
      // Create calendar grid
      const calendarGrid = document.getElementById('calendarGrid');
      calendarGrid.innerHTML = '';
      
      // Days of week headers
      const daysOfWeek = calendarNames?.days || ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
      daysOfWeek.forEach(dayName => {
        const header = document.createElement('div');
        header.style.fontSize = '12px';
        header.style.color = '#8e8e93';
        header.style.textAlign = 'center';
        header.textContent = dayName.substring(0, 3);
        calendarGrid.appendChild(header);
      });
      
      // Calculate days in month (simplified - 30 days)
      const daysInMonth = 30;
      const firstDayOfWeek = 0; // Start on first day
      
      // Empty cells for alignment
      for (let i = 0; i < firstDayOfWeek; i++) {
        const emptyDay = document.createElement('div');
        calendarGrid.appendChild(emptyDay);
      }
      
      // Days of month
      for (let i = 1; i <= daysInMonth; i++) {
        const dayEl = document.createElement('div');
        dayEl.className = 'calendar-day';
        if (i === day) {
          dayEl.classList.add('active');
        }
        dayEl.textContent = i;
        calendarGrid.appendChild(dayEl);
      }
      
      // Today's schedule
      const scheduleEl = document.getElementById('todaySchedule');
      const schedule = data.currentRoleplay?.ChaseSchedule;
      if (schedule) {
        try {
          const scheduleData = typeof schedule === 'string' ? JSON.parse(schedule) : schedule;
          const todaySchedule = scheduleData[daysOfWeek[new Date().getDay()]] || {};
          
          let scheduleHTML = '<div style="display: grid; gap: 12px;">';
          const timeSlots = ['Morning', 'Afternoon', 'Evening', 'Night'];
          
          timeSlots.forEach(slot => {
            const activity = todaySchedule[slot] || 'Free time';
            const isCurrentTime = slot === timeOfDay;
            scheduleHTML += `
              <div style="padding: 12px; background-color: ${isCurrentTime ? '#007aff' : document.body.classList.contains('light-mode') ? '#f5f5f7' : '#2c2c2e'}; 
                          color: ${isCurrentTime ? 'white' : 'inherit'}; border-radius: 8px;">
                <strong>${slot}:</strong> ${activity}
              </div>
            `;
          });
          
          scheduleHTML += '</div>';
          scheduleEl.innerHTML = scheduleHTML;
        } catch (e) {
          scheduleEl.innerHTML = '<p>No schedule available</p>';
        }
      }
    }
  } catch (error) {
    console.error('Error loading calendar:', error);
  }
}

// Journal/Quests
async function loadJournal() {
  try {
    const data = await fetchJson(`/universal/get_aggregated_roleplay_context?conversation_id=${AppState.currentConvId}&player_name=Chase`);
    
    // Display active quests
    const questsContainer = document.getElementById('activeQuests');
    if (data.quests && data.quests.length > 0) {
      questsContainer.innerHTML = data.quests.map(quest => `
        <div class="quest-card">
          <div class="quest-title">${quest.quest_name}</div>
          <div class="quest-description">${quest.progress_detail || 'No details available'}</div>
          <div class="quest-progress">
            <span>Status: ${quest.status}</span>
            ${quest.quest_giver ? `<span>Given by: ${quest.quest_giver}</span>` : ''}
          </div>
        </div>
      `).join('');
    } else {
      questsContainer.innerHTML = '<p>No active quests</p>';
    }
    
    // Journal entries would go here - need an endpoint for this
    const journalContainer = document.getElementById('journalEntries');
    journalContainer.innerHTML = '<p>Journal entries coming soon...</p>';
    
  } catch (error) {
    console.error('Error loading journal:', error);
  }
}

// Inventory
async function loadInventory() {
  try {
    const data = await fetchJson(`/universal/get_aggregated_roleplay_context?conversation_id=${AppState.currentConvId}&player_name=Chase`);
    
    const inventoryGrid = document.getElementById('inventoryGrid');
    if (data.inventory && data.inventory.length > 0) {
      inventoryGrid.innerHTML = data.inventory.map(item => {
        // Map item categories to emoji icons
        const iconMap = {
          'weapon': '⚔️',
          'armor': '🛡️',
          'consumable': '🧪',
          'key': '🗝️',
          'tool': '🔧',
          'misc': '📦',
          'food': '🍎',
          'drink': '🥤'
        };
        
        const icon = iconMap[item.category?.toLowerCase()] || '📦';
        
        return `
          <div class="inventory-item" title="${item.item_description || 'No description'}">
            <div class="inventory-icon">${icon}</div>
            <div class="inventory-name">${item.item_name}</div>
            <div class="inventory-quantity">x${item.quantity || 1}</div>
          </div>
        `;
      }).join('');
    } else {
      inventoryGrid.innerHTML = '<p style="grid-column: 1/-1; text-align: center;">Your inventory is empty</p>';
    }
  } catch (error) {
    console.error('Error loading inventory:', error);
  }
}

// Social Links - Only show introduced NPCs
async function loadSocialLinks() {
  try {
    const data = await fetchJson(`/universal/get_aggregated_roleplay_context?conversation_id=${AppState.currentConvId}&player_name=Chase`);
    
    const socialContainer = document.getElementById('socialLinks');
    if (data.socialLinks && data.socialLinks.length > 0 && data.npcStats) {
      // Get introduced NPCs
      const introducedNPCs = data.npcStats.filter(npc => npc.introduced === 1);
      const introducedNPCIds = introducedNPCs.map(npc => npc.npc_id);
      
      // Filter for player relationships with introduced NPCs
      const playerLinks = data.socialLinks.filter(link => {
        const isEntity1Player = link.entity1_type === 'player' && link.entity1_id == AppState.userId;
        const isEntity2Player = link.entity2_type === 'player' && link.entity2_id == AppState.userId;
        
        if (isEntity1Player && link.entity2_type === 'npc') {
          return introducedNPCIds.includes(link.entity2_id);
        } else if (isEntity2Player && link.entity1_type === 'npc') {
          return introducedNPCIds.includes(link.entity1_id);
        }
        return false;
      });
      
      if (playerLinks.length > 0) {
        socialContainer.innerHTML = playerLinks.map(link => {
          // Determine which entity is the NPC
          const isEntity1Player = link.entity1_type === 'player';
          const npcType = isEntity1Player ? link.entity2_type : link.entity1_type;
          const npcId = isEntity1Player ? link.entity2_id : link.entity1_id;
          
          // Find NPC name from data
          let npcName = 'Unknown';
          let npcRole = '';
          if (npcType === 'npc') {
            const npc = introducedNPCs.find(n => n.npc_id == npcId);
            if (npc) {
              npcName = npc.npc_name;
              npcRole = npc.role || '';
            }
          }
          
          // Color code by relationship type
          const typeColors = {
            'friendly': '#34c759',
            'romantic': '#ff2d55',
            'professional': '#007aff',
            'rival': '#ff9500',
            'neutral': '#8e8e93'
          };
          
          const color = typeColors[link.link_type] || '#8e8e93';
          
          return `
            <div class="social-link-card">
              <div class="social-link-info">
                <div class="social-link-name">${npcName}</div>
                <div class="social-link-type" style="color: ${color};">${link.link_type}${npcRole ? ` · ${npcRole}` : ''}</div>
              </div>
              <div class="social-link-level">
                <span style="font-size: 24px; font-weight: 600;">${link.link_level}</span>
                <div class="progress-bar" style="width: 60px;">
                  <div class="progress-fill" style="width: ${link.link_level}%; background: ${color};"></div>
                </div>
              </div>
            </div>
          `;
        }).join('');
      } else {
        socialContainer.innerHTML = '<p>No relationships established yet</p>';
      }
    } else {
      socialContainer.innerHTML = '<p>No relationships established yet</p>';
    }
  } catch (error) {
    console.error('Error loading social links:', error);
  }
}

// Codex - New feature for lore/NPCs/locations reference
async function loadCodex() {
  try {
    const data = await fetchJson(`/universal/get_aggregated_roleplay_context?conversation_id=${AppState.currentConvId}&player_name=Chase`);
    
    // Load Characters (introduced NPCs only)
    const charactersContainer = document.getElementById('codexCharacters');
    if (data.npcStats) {
      const introducedNPCs = data.npcStats.filter(npc => npc.introduced === 1);
      
      if (introducedNPCs.length > 0) {
        charactersContainer.innerHTML = introducedNPCs.map(npc => `
          <div class="codex-entry">
            <div class="codex-entry-type">NPC</div>
            <div class="codex-entry-title">${npc.npc_name}</div>
            <div class="codex-entry-description">
              ${npc.role ? `Role: ${npc.role}<br>` : ''}
              ${npc.description || 'No description available yet.'}
            </div>
          </div>
        `).join('');
      } else {
        charactersContainer.innerHTML = '<p style="grid-column: 1/-1;">No characters discovered yet</p>';
      }
    }
    
    // Load Locations (if available in currentRoleplay)
    const locationsContainer = document.getElementById('codexLocations');
    if (data.currentRoleplay && data.currentRoleplay.discovered_locations) {
      try {
        const locations = typeof data.currentRoleplay.discovered_locations === 'string' 
          ? JSON.parse(data.currentRoleplay.discovered_locations) 
          : data.currentRoleplay.discovered_locations;
        
        if (locations && locations.length > 0) {
          locationsContainer.innerHTML = locations.map(location => `
            <div class="codex-entry">
              <div class="codex-entry-type">LOCATION</div>
              <div class="codex-entry-title">${location.name}</div>
              <div class="codex-entry-description">${location.description || 'A location in the world.'}</div>
            </div>
          `).join('');
        } else {
          locationsContainer.innerHTML = '<p style="grid-column: 1/-1;">No locations discovered yet</p>';
        }
      } catch (e) {
        locationsContainer.innerHTML = '<p style="grid-column: 1/-1;">No locations discovered yet</p>';
      }
    } else {
      locationsContainer.innerHTML = '<p style="grid-column: 1/-1;">No locations discovered yet</p>';
    }
    
    // Load Lore (if available)
    const loreContainer = document.getElementById('codexLore');
    if (data.currentRoleplay && data.currentRoleplay.lore_entries) {
      try {
        const loreEntries = typeof data.currentRoleplay.lore_entries === 'string' 
          ? JSON.parse(data.currentRoleplay.lore_entries) 
          : data.currentRoleplay.lore_entries;
        
        if (loreEntries && loreEntries.length > 0) {
          loreContainer.innerHTML = loreEntries.map(entry => `
            <div class="codex-entry">
              <div class="codex-entry-type">LORE</div>
              <div class="codex-entry-title">${entry.title}</div>
              <div class="codex-entry-description">${entry.content}</div>
            </div>
          `).join('');
        } else {
          loreContainer.innerHTML = '<p style="grid-column: 1/-1;">No lore discovered yet</p>';
        }
      } catch (e) {
        loreContainer.innerHTML = '<p style="grid-column: 1/-1;">No lore discovered yet</p>';
      }
    } else {
      loreContainer.innerHTML = '<p style="grid-column: 1/-1;">No lore discovered yet</p>';
    }
    
  } catch (error) {
    console.error('Error loading codex:', error);
  }
}

// Update conversation button styles
function updateConversationButtons() {
  const buttons = document.querySelectorAll('#convList button');
  buttons.forEach(btn => {
    if (btn.dataset.convId == AppState.currentConvId) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
}

// Override the existing selectConversation to update UI
const originalSelectConversation = window.selectConversation;
window.selectConversation = async function(convId) {
  await originalSelectConversation(convId);
  updateConversationButtons();
  
  // Load data for active tab
  const activeTab = document.querySelector('.tab.active');
  if (activeTab) {
    await loadTabData(activeTab.dataset.tab);
  }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
  // Initialize admin UI first
  initializeAdminUI();
  
  // Add this after existing initialization
  setTimeout(() => {
    initializeTabs();
    
    // Update theme button text
    const themeBtn = document.getElementById('toggleThemeBtn');
    if (themeBtn) {
      themeBtn.addEventListener('click', () => {
        document.body.classList.toggle('light-mode');
        const isLight = document.body.classList.contains('light-mode');
        themeBtn.textContent = isLight ? '🌙 Dark Mode' : '☀️ Light Mode';
        
        // Save preference
        localStorage.setItem('theme', isLight ? 'light' : 'dark');
      });
      
      // Load saved theme preference
      const savedTheme = localStorage.getItem('theme');
      if (savedTheme === 'light') {
        document.body.classList.add('light-mode');
        themeBtn.textContent = '🌙 Dark Mode';
      } else {
        // Dark mode is default, ensure light-mode class is removed
        document.body.classList.remove('light-mode');
        themeBtn.textContent = '☀️ Light Mode';
      }
    }
    
    // Fix dropdown toggle
    const newGameBtn = document.getElementById('newGameBtn');
    const dropdown = document.getElementById('newGameDropdown');
    
    if (newGameBtn && dropdown) {
      newGameBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        dropdown.style.display = dropdown.style.display === 'block' ? 'none' : 'block';
      });
      
      // Close dropdown when clicking outside
      document.addEventListener('click', () => {
        dropdown.style.display = 'none';
      });
    }
  }, 100);
});
