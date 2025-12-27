/**
 * Global state management with Zustand
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Chat store - manages conversations and messages
export const useChatStore = create(
  persist(
    (set, get) => ({
      // Conversations
      conversations: [],
      activeConversationId: null,
      isLoading: false,
      
      // Load chats from backend
      loadChats: async (username) => {
        if (!username) return;
        console.log('[ChatStore] loadChats called for:', username);
        set({ isLoading: true });
        try {
          const response = await fetch(`${API_URL}/chats/${username}`);
          console.log('[ChatStore] loadChats response status:', response.status);
          if (response.ok) {
            const data = await response.json();
            console.log('[ChatStore] loadChats data:', data);
            const { activeConversationId } = get();
            
            const loadedConversations = data.chats.map(chat => ({
              id: chat.id,
              title: chat.title,
              messages: [], // Messages loaded on demand
              createdAt: chat.createdAt,
              updatedAt: chat.updatedAt,
              messageCount: chat.messageCount,
              synced: true,
            }));
            
            // Check if saved activeConversationId exists in loaded chats
            const activeExists = loadedConversations.some(c => c.id === activeConversationId);
            
            set({
              conversations: loadedConversations,
              // Keep active ID if it exists, otherwise select first chat or null
              activeConversationId: activeExists 
                ? activeConversationId 
                : (loadedConversations[0]?.id || null),
            });
          }
        } catch (error) {
          console.error('Failed to load chats:', error);
        } finally {
          set({ isLoading: false });
        }
      },
      
      // Load messages for a specific chat
      loadChatMessages: async (username, sessionId) => {
        if (!username || !sessionId) return;
        try {
          const response = await fetch(`${API_URL}/chats/${username}/${sessionId}`);
          if (response.ok) {
            const data = await response.json();
            set((state) => ({
              conversations: state.conversations.map(conv =>
                conv.id === sessionId
                  ? {
                      ...conv,
                      messages: data.messages.map(msg => ({
                        id: msg.id,
                        type: msg.role,
                        content: msg.content,
                        timestamp: msg.timestamp,
                        execution: msg.executionData, // Map to 'execution' for MessageBubble compatibility
                      })),
                    }
                  : conv
              ),
            }));
          }
        } catch (error) {
          console.error('Failed to load messages:', error);
        }
      },
      
      // Create new conversation
      createConversation: async (username, title = 'New Chat') => {
        // Create locally first for instant feedback
        const tempId = `conv_${Date.now()}`;
        const conversation = {
          id: tempId,
          title,
          messages: [],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          synced: false,
        };
        
        set((state) => ({
          conversations: [conversation, ...state.conversations],
          activeConversationId: tempId,
        }));
        
        // Sync with backend if username provided
        if (username) {
          try {
            const response = await fetch(`${API_URL}/chats`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ username, title }),
            });
            if (response.ok) {
              const data = await response.json();
              // Update local conversation with server ID
              set((state) => ({
                conversations: state.conversations.map(conv =>
                  conv.id === tempId
                    ? { ...conv, id: data.chat.id, synced: true }
                    : conv
                ),
                activeConversationId: data.chat.id,
              }));
              return data.chat.id;
            }
          } catch (error) {
            console.error('Failed to create chat on server:', error);
          }
        }
        
        return tempId;
      },
      
      // Set active conversation
      setActiveConversation: (id) => set({ activeConversationId: id }),
      
      // Add message to active conversation
      addMessage: async (message, username) => {
        // Always get fresh state
        const { activeConversationId, conversations } = get();
        
        console.log('[addMessage] activeConversationId:', activeConversationId, 'username:', username);
        
        // If no active conversation, skip (shouldn't happen if createConversation was called)
        if (!activeConversationId) {
          console.warn('addMessage called with no activeConversationId');
          return;
        }
        
        const msgId = `msg_${Date.now()}`;
        const activeConv = conversations.find(c => c.id === activeConversationId);
        const isFirstUserMessage = activeConv?.messages.length === 0 && message.type === 'user';
        const newTitle = isFirstUserMessage 
          ? message.content.slice(0, 50) + (message.content.length > 50 ? '...' : '')
          : null;
        
        // Normalize message format - use 'execution' for consistency with MessageBubble
        const normalizedMessage = {
          id: msgId,
          type: message.type,
          content: message.content,
          timestamp: message.timestamp || new Date(),
          execution: message.executionData || message.execution || null,
        };
        
        // Update local state
        set((state) => ({
          conversations: state.conversations.map((conv) =>
            conv.id === activeConversationId
              ? {
                  ...conv,
                  messages: [...conv.messages, normalizedMessage],
                  updatedAt: new Date().toISOString(),
                  title: newTitle || conv.title,
                }
              : conv
          ),
        }));
        
        // Sync with backend if conversation is synced (has chat_ prefix)
        console.log('[addMessage] Checking sync:', { username, activeConversationId, startsWithChat: activeConversationId?.startsWith('chat_') });
        if (username && activeConversationId.startsWith('chat_')) {
          console.log('[addMessage] Syncing to backend...');
          try {
            const response = await fetch(`${API_URL}/chats/${activeConversationId}/messages`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                role: message.type,
                content: message.content,
                execution_data: message.executionData || message.execution || null,
              }),
            });
            console.log('[addMessage] Response status:', response.status);
            if (response.ok) {
              const data = await response.json();
              console.log('[addMessage] Response data:', data);
              // Update title if changed by backend
              if (data.chatTitle && data.chatTitle !== 'New Chat') {
                set((state) => ({
                  conversations: state.conversations.map(conv =>
                    conv.id === activeConversationId
                      ? { ...conv, title: data.chatTitle }
                      : conv
                  ),
                }));
              }
            }
          } catch (error) {
            console.error('Failed to save message:', error);
          }
        }
      },
      
      // Update last message (for streaming)
      updateLastMessage: (updates) => {
        const { activeConversationId, conversations } = get();
        if (!activeConversationId) return;
        
        set({
          conversations: conversations.map((conv) =>
            conv.id === activeConversationId
              ? {
                  ...conv,
                  messages: conv.messages.map((msg, idx) =>
                    idx === conv.messages.length - 1 ? { ...msg, ...updates } : msg
                  ),
                  updatedAt: new Date().toISOString(),
                }
              : conv
          ),
        });
      },
      
      // Delete conversation
      deleteConversation: async (id, username) => {
        const { conversations, activeConversationId } = get();
        const newConversations = conversations.filter((c) => c.id !== id);
        set({
          conversations: newConversations,
          activeConversationId:
            activeConversationId === id
              ? newConversations[0]?.id || null
              : activeConversationId,
        });
        
        // Delete from backend
        if (username && id.startsWith('chat_')) {
          try {
            await fetch(`${API_URL}/chats/${id}`, { method: 'DELETE' });
          } catch (error) {
            console.error('Failed to delete chat:', error);
          }
        }
      },
      
      // Clear all conversations
      clearAll: () => set({ conversations: [], activeConversationId: null }),
      
      // Get active conversation
      getActiveConversation: () => {
        const { conversations, activeConversationId } = get();
        return conversations.find((c) => c.id === activeConversationId);
      },
    }),
    {
      name: 'magentic-chat-storage',
      partialize: (state) => ({
        // Don't persist conversations - use backend as source of truth
        // Only persist the active conversation ID for UX continuity
        activeConversationId: state.activeConversationId,
      }),
    }
  )
);

// UI Store - manages UI state
export const useUIStore = create(
  persist(
    (set) => ({
      // Sidebar
      sidebarOpen: true,
      sidebarWidth: 280,
      
      // Theme
      theme: 'dark', // 'dark' | 'light' | 'system'
      
      // View settings
      showExecutionDetails: true,
      showAgentGraph: false,
      compactMode: false,
      
      // Expanded steps
      expandedSteps: new Set(),
      
      // Actions
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      setSidebarWidth: (width) => set({ sidebarWidth: width }),
      
      setTheme: (theme) => {
        // Update DOM
        if (theme === 'dark') {
          document.documentElement.classList.add('dark');
        } else {
          document.documentElement.classList.remove('dark');
        }
        set({ theme });
      },
      toggleExecutionDetails: () => set((state) => ({ showExecutionDetails: !state.showExecutionDetails })),
      toggleAgentGraph: () => set((state) => ({ showAgentGraph: !state.showAgentGraph })),
      toggleCompactMode: () => set((state) => ({ compactMode: !state.compactMode })),
      
      toggleStep: (key) => set((state) => {
        const newSet = new Set(state.expandedSteps);
        if (newSet.has(key)) newSet.delete(key);
        else newSet.add(key);
        return { expandedSteps: newSet };
      }),
    }),
    {
      name: 'magentic-ui-storage',
      partialize: (state) => ({
        theme: state.theme,
        showExecutionDetails: state.showExecutionDetails,
        compactMode: state.compactMode,
        sidebarWidth: state.sidebarWidth,
      }),
      onRehydrateStorage: () => (state) => {
        // Apply theme to DOM after hydration from localStorage
        if (state?.theme === 'dark') {
          document.documentElement.classList.add('dark');
        } else {
          document.documentElement.classList.remove('dark');
        }
      },
    }
  )
);

// Execution Store - manages current execution state
export const useExecutionStore = create((set) => ({
  currentExecution: null,
  isExecuting: false,
  
  setExecution: (execution) => set({ currentExecution: execution }),
  setIsExecuting: (isExecuting) => set({ isExecuting }),
  clearExecution: () => set({ currentExecution: null, isExecuting: false }),
  
  // Update agent status within execution
  updateAgentStatus: (agentId, status) => set((state) => {
    if (!state.currentExecution) return state;
    return {
      currentExecution: {
        ...state.currentExecution,
        agents: state.currentExecution.agents?.map((a) =>
          a.agent_id === agentId ? { ...a, ...status } : a
        ) || [],
      },
    };
  }),
}));

// Connection Store
export const useConnectionStore = create((set) => ({
  isConnected: false,
  reconnecting: false,
  lastError: null,
  
  setConnected: (isConnected) => set({ isConnected, reconnecting: false }),
  setReconnecting: (reconnecting) => set({ reconnecting }),
  setError: (error) => set({ lastError: error }),
}));
