// npc_monitor.js
$(document).ready(function() {
    const userId = $('#user-id');
    const conversationId = $('#conversation-id');
    const npcsContainer = $('#npcs-container');
    const loadNpcsBtn = $('#load-npcs');
    const refreshBtn = $('#refresh-data');
    const batchProcessBtn = $('#batch-process');
    const loading = $('#loading');
    const noNpcs = $('#no-npcs');

    let loadedNpcIds = [];

    // Load NPCs button click handler
    loadNpcsBtn.click(function() {
        loadNpcs();
    });

    // Refresh data button click handler
    refreshBtn.click(function() {
        refreshNpcData();
    });

    // Batch process button click handler
    batchProcessBtn.click(function() {
        processBatch();
    });

    // Load NPCs for the current user and conversation
    function loadNpcs() {
        const uid = userId.val();
        const cid = conversationId.val();

        if (!uid || !cid) {
            alert('Please enter User ID and Conversation ID.');
            return;
        }

        loading.show();
        npcsContainer.empty();
        loadedNpcIds = [];

        // Fetch NPCs from the database
        $.ajax({
            url: `/npcs/api/get-all?user_id=${uid}&conversation_id=${cid}`,
            type: 'GET',
            success: function(response) {
                loading.hide();

                if (response.npcs && response.npcs.length > 0) {
                    response.npcs.forEach(npc => {
                        loadedNpcIds.push(npc.npc_id);
                        getNpcLearningStatus(npc.npc_id);
                    });

                    refreshBtn.prop('disabled', false);
                    batchProcessBtn.prop('disabled', false);
                    noNpcs.hide();
                } else {
                    noNpcs.show();
                    refreshBtn.prop('disabled', true);
                    batchProcessBtn.prop('disabled', true);
                }
            },
            error: function(xhr) {
                loading.hide();
                alert('Error loading NPCs: ' + xhr.responseText);
            }
        });
    }

    // Get learning status for a specific NPC
    function getNpcLearningStatus(npcId) {
        const uid = userId.val();
        const cid = conversationId.val();

        $.ajax({
            url: `/api/npc/learning/status/${npcId}?user_id=${uid}&conversation_id=${cid}`,
            type: 'GET',
            success: function(response) {
                displayNpcCard(response);
            },
            error: function(xhr) {
                console.error('Error getting NPC learning status:', xhr.responseText);
            }
        });
    }

    // Display NPC card with learning data
    function displayNpcCard(npcData) {
        const template = document.getElementById('npc-card-template');
        const clone = document.importNode(template.content, true);

        // Set NPC info
        $(clone).find('.npc-name').text(npcData.npc_name);
        $(clone).find('.intensity-value').text(npcData.stats.intensity);

        // Set button data attributes
        $(clone).find('.process-npc-btn').attr('data-npc-id', npcData.npc_id);
        $(clone).find('.trigger-learning-btn').attr('data-npc-id', npcData.npc_id);

        // Set progress bars
        $(clone).find('.intensity-bar').css('width', `${npcData.stats.intensity}%`).text(npcData.stats.intensity);
        $(clone).find('.dominance-bar').css('width', `${npcData.stats.dominance}%`).text(npcData.stats.dominance);
        $(clone).find('.cruelty-bar').css('width', `${npcData.stats.cruelty}%`).text(npcData.stats.cruelty);
        $(clone).find('.aggression-bar').css('width', `${npcData.stats.aggression}%`).text(npcData.stats.aggression);
        $(clone).find('.manipulativeness-bar').css('width', `${npcData.stats.manipulativeness}%`).text(npcData.stats.manipulativeness);

        // Color the intensity badge based on value
        const badge = $(clone).find('.badge');
        if (npcData.stats.intensity >= 70) {
            badge.addClass('bg-danger');
        } else if (npcData.stats.intensity >= 40) {
            badge.addClass('bg-warning');
        } else {
            badge.addClass('bg-success');
        }

        // Add memories
        const memoriesList = $(clone).find('.memories-list');
        if (npcData.learning_memories && npcData.learning_memories.length > 0) {
            npcData.learning_memories.forEach(memory => {
                const memoryTemplate = document.getElementById('memory-item-template');
                const memoryClone = document.importNode(memoryTemplate.content, true);

                $(memoryClone).find('.memory-text').text(memory.memory_text);
                $(memoryClone).find('.memory-created').text(`Created: ${new Date(memory.created_at).toLocaleString()}`);

                memoriesList.append(memoryClone);
            });
        } else {
            memoriesList.html('<p class="text-muted">No learning memories yet.</p>');
        }

        // Add the card to the container
        npcsContainer.append(clone);

        // Add event listeners for the buttons dynamically *after* they are added to the DOM
        // Use event delegation for buttons inside the template for better performance and to handle dynamically added elements.
        // However, since we re-bind '.process-npc-btn' and '.trigger-learning-btn' after each card is added,
        // the original method of direct binding is fine, but could be optimized if many cards are added.
        // For simplicity, keeping existing logic for now as it works.

        // Re-bind click events for dynamically added buttons to avoid duplicate handlers
        // or handlers on elements that might have been removed and re-added.
        // A more robust way is event delegation from a static parent, e.g., npcsContainer.
        npcsContainer.off('click', '.process-npc-btn').on('click', '.process-npc-btn', function() {
            const npcId = $(this).data('npc-id');
            processNpcLearning(npcId);
        });

        npcsContainer.off('click', '.trigger-learning-btn').on('click', '.trigger-learning-btn', function() {
            const npcId = $(this).data('npc-id');
            $('#trigger-npc-id').val(npcId);
            // Note: data-bs-toggle="modal" and data-bs-target="#triggerModal" are handled by Bootstrap
        });
    }

    // Process NPC learning
    function processNpcLearning(npcId) {
        const uid = userId.val();
        const cid = conversationId.val();

        $.ajax({
            url: `/api/npc/learning/process/${npcId}`,
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                user_id: parseInt(uid),
                conversation_id: parseInt(cid)
            }),
            success: function(response) {
                alert(`Learning processed for NPC ${npcId}. Refreshing specific card...`);
                // To refresh only the specific card, we might need to find it and update it
                // For now, let's re-fetch its status.
                // First, remove the old card to avoid duplicates if getNpcLearningStatus re-appends
                $(`.process-npc-btn[data-npc-id="${npcId}"]`).closest('.col-md-6').remove();
                getNpcLearningStatus(npcId);
            },
            error: function(xhr) {
                alert('Error processing learning: ' + xhr.responseText);
            }
        });
    }

    // Refresh all loaded NPCs
    function refreshNpcData() {
        if (loadedNpcIds.length === 0) {
            alert('No NPCs loaded to refresh.');
            return;
        }

        loading.show(); // Show loading indicator
        npcsContainer.empty(); // Clear existing cards
        // Re-fetch status for all loaded NPCs
        const promises = loadedNpcIds.map(npcId => {
            return new Promise((resolve, reject) => {
                const uid = userId.val();
                const cid = conversationId.val();
                $.ajax({
                    url: `/api/npc/learning/status/${npcId}?user_id=${uid}&conversation_id=${cid}`,
                    type: 'GET',
                    success: function(response) {
                        displayNpcCard(response);
                        resolve();
                    },
                    error: function(xhr) {
                        console.error(`Error refreshing NPC ${npcId} learning status:`, xhr.responseText);
                        reject(xhr);
                    }
                });
            });
        });

        Promise.all(promises)
            .then(() => loading.hide())
            .catch(() => loading.hide()); // Hide loading even if some fail
    }

    // Process batch learning for all NPCs
    function processBatch() {
        const uid = userId.val();
        const cid = conversationId.val();

        if (loadedNpcIds.length === 0) {
            alert('No NPCs loaded to process.');
            return;
        }

        $.ajax({
            url: '/api/npc/learning/batch-process',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                user_id: parseInt(uid),
                conversation_id: parseInt(cid),
                npc_ids: loadedNpcIds
            }),
            success: function(response) {
                alert(`Batch processing completed for ${response.npc_count} NPCs. Refreshing data...`);
                refreshNpcData(); // Refresh to show updated states
            },
            error: function(xhr) {
                alert('Error in batch processing: ' + xhr.responseText);
            }
        });
    }

    // Trigger learning event (Modal submit button)
    $('#submit-trigger').click(function() {
        const npcId = $('#trigger-npc-id').val();
        const triggerType = $('#trigger-type').val();
        const summary = $('#trigger-summary').val();
        const uid = userId.val();
        const cid = conversationId.val();

        if (!summary) {
            alert('Please enter a summary for the trigger event.');
            return;
        }

        $.ajax({
            url: `/api/npc/learning/trigger/${npcId}`,
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                user_id: parseInt(uid),
                conversation_id: parseInt(cid),
                trigger_type: triggerType,
                trigger_details: {
                    summary: summary
                }
            }),
            success: function(response) {
                $('#triggerModal').modal('hide'); // Hide the modal
                alert(`Learning trigger processed for NPC ${npcId}. ${response.result.message || ''}. Refreshing card...`);
                // Refresh the specific NPC card
                $(`.process-npc-btn[data-npc-id="${npcId}"]`).closest('.col-md-6').remove();
                getNpcLearningStatus(npcId);
            },
            error: function(xhr) {
                alert('Error triggering learning event: ' + xhr.responseText);
            }
        });
    });
});
