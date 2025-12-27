/**
 * Main App component - Magentic chat interface v3.0
 * Redesigned with animated UI and agent visualization
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import { useAuth } from './contexts/AuthContext';
import { useWebSocket, processWebSocketMessage } from './hooks';
import {
  Header,
  EmptyState,
  LoginModal,
  ProfileModal,
  Sidebar,
  MessageBubble,
  EnhancedChatInput,
  AgentFlowGraph,
  SettingsPanel,
  ExecutionProgress,
} from './components';
import { useUIStore, useExecutionStore, useConnectionStore, useChatStore } from './store';

function App() {
  const { user, isAuthenticated, isGuest, loading } = useAuth();
  
  // Memoize user data to prevent unnecessary re-renders
  const stableUser = useMemo(() => ({
    username: user?.username,
    display_name: user?.display_name,
    avatar_emoji: user?.avatar_emoji,
  }), [user?.username, user?.display_name, user?.avatar_emoji]);
  
  // Chat store for persistence - only get what we need to avoid re-renders
  const activeConversationId = useChatStore((state) => state.activeConversationId);
  const loadChats = useChatStore((state) => state.loadChats);
  const createConversation = useChatStore((state) => state.createConversation);
  const addMessage = useChatStore((state) => state.addMessage);
  
  // Local state for current session messages
  const [messages, setMessages] = useState([]);
  const [currentExecution, setCurrentExecution] = useState(null);
  
  // Zustand stores for UI only
  const {
    sidebarOpen,
    settingsOpen,
    showExecutionDetails,
    showAgentFlow,
    toggleSidebar,
    toggleSettings,
    toggleExecutionDetails,
    toggleAgentFlow,
    theme,
  } = useUIStore();
  
  const { setExecution, clearExecution } = useExecutionStore();
  const { setConnected } = useConnectionStore();
  
  // Local UI state
  const [showProfile, setShowProfile] = useState(false);
  const [showLogin, setShowLogin] = useState(false);
  const [expandedSteps, setExpandedSteps] = useState(new Set());
  
  // Refs
  const messagesEndRef = useRef(null);
  const executionRef = useRef(null);

  // Keep execution ref in sync with state
  useEffect(() => {
    executionRef.current = currentExecution;
  }, [currentExecution]);

  // Message handler for WebSocket
  const handleWebSocketMessage = useCallback((data) => {
    processWebSocketMessage(data, setCurrentExecution, setMessages, executionRef);
    
    // Update execution store for visualization
    if (data.type === 'agent_start' || data.type === 'agent_end' || data.type === 'tool_start') {
      setExecution(data);
    }
    
    // Save assistant response to chat store when complete (for backend persistence)
    if (data.type === 'complete' && data.data?.output) {
      const username = user?.username || 'guest';
      // Get execution data and ensure token_usage is included
      const executionData = executionRef.current
        ? JSON.parse(JSON.stringify(executionRef.current))
        : null;
      
      // Add token usage from complete event
      if (executionData && data.data.token_usage) {
        executionData.token_usage = data.data.token_usage;
      }
      
      addMessage({
        type: 'assistant',
        content: data.data.output,
        execution: executionData,
        timestamp: new Date(),
      }, username);
    }
  }, [setExecution, user, addMessage]);

  // WebSocket connection
  const { isConnected, sendMessage } = useWebSocket(
    user,
    isAuthenticated,
    handleWebSocketMessage
  );

  // Sync connection state
  useEffect(() => {
    setConnected(isConnected);
  }, [isConnected, setConnected]);

  // Show login modal if not authenticated (with small delay to prevent flash)
  useEffect(() => {
    if (!loading && !isAuthenticated) {
      // Small delay to ensure session restore has completed
      const timer = setTimeout(() => {
        setShowLogin(true);
      }, 100);
      return () => clearTimeout(timer);
    } else {
      setShowLogin(false);
    }
  }, [loading, isAuthenticated]);
  
  // Load chats when user logs in
  useEffect(() => {
    if (isAuthenticated && user?.username) {
      console.log('[App] Loading chats for user:', user.username);
      loadChats(user.username);
    }
  }, [isAuthenticated, user?.username, loadChats]);
  
  // Track previous conversation ID to detect switches
  const prevConversationIdRef = useRef(activeConversationId);
  
  // Load messages from backend when needed
  const loadChatMessages = useChatStore((state) => state.loadChatMessages);
  const conversations = useChatStore((state) => state.conversations);
  
  // Sync messages when conversation changes OR when conversations are first loaded
  useEffect(() => {
    // Skip if no active conversation
    if (!activeConversationId) {
      setMessages([]);
      return;
    }
    
    // Find the active conversation
    const activeConv = conversations.find(c => c.id === activeConversationId);
    
    // Skip if conversation not found (might be loading)
    if (!activeConv) {
      return;
    }
    
    // Check if this is a conversation switch or initial load
    const isSwitch = prevConversationIdRef.current !== activeConversationId;
    if (isSwitch) {
      prevConversationIdRef.current = activeConversationId;
      setCurrentExecution(null); // Reset execution state for new/switched chat
    }
    
    // If this is a synced conversation with messages that haven't been loaded, fetch from backend
    if (activeConv.synced && activeConv.messageCount > 0 && activeConv.messages.length === 0) {
      const username = user?.username || 'guest';
      loadChatMessages(username, activeConversationId);
    } else {
      // Use messages from store (either already loaded or local)
      setMessages(activeConv.messages || []);
    }
  }, [activeConversationId, conversations, user, loadChatMessages]);

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentExecution, scrollToBottom]);

  // Handle send message
  const handleSend = useCallback(async (content) => {
    if (!content.trim() || !isConnected) return;
    
    const username = user?.username || 'guest';
    
    // Create a new conversation if none active
    let currentConvId = activeConversationId;
    if (!currentConvId) {
      currentConvId = await createConversation(username);
      // Wait a tick for state to update
      await new Promise(resolve => setTimeout(resolve, 50));
    }

    const userMessage = {
      type: 'user',
      content,
      timestamp: new Date(),
    };

    // Add user message locally
    setMessages((prev) => [...prev, userMessage]);
    
    // Save to store (which syncs to backend)
    await addMessage(userMessage, username);

    // Set immediate execution state with loading indicator (single box for progress)
    setCurrentExecution({
      stage: 'initializing',
      stageMessage: 'Processing your query...',
      isLoading: true,
      agents: [],
      plan: null,
    });

    // Send to WebSocket with session_id for tracking
    sendMessage({ query: content, session_id: currentConvId });
  }, [isConnected, sendMessage, user, activeConversationId, createConversation, addMessage]);

  // Handle stop execution
  const handleStop = useCallback(() => {
    // Send stop signal - let the backend response handler clear the execution
    sendMessage({ type: 'stop' });
    // Don't clear execution here - wait for 'stopped' message from backend
  }, [sendMessage]);

  // Toggle step expansion
  const toggleStep = useCallback((key) => {
    setExpandedSteps((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(key)) {
        newSet.delete(key);
      } else {
        newSet.add(key);
      }
      return newSet;
    });
  }, []);

  // Sync theme on mount (handles hydration from localStorage)
  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  const isProcessing = !!currentExecution;

  return (
    <div className="flex h-screen overflow-hidden transition-colors duration-200 bg-slate-50 dark:bg-gray-950">
      
      {/* Toast notifications */}
      <Toaster 
        position="top-right"
        toastOptions={{
          className: 'bg-white dark:bg-gray-800 text-slate-700 dark:text-white shadow-lg border border-slate-200/50 dark:border-gray-700',
          duration: 4000,
        }}
      />
      
      {/* Sidebar */}
      <Sidebar 
        isOpen={sidebarOpen} 
        onClose={() => toggleSidebar()}
        onOpenSettings={() => toggleSettings()}
      />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <Header
          user={stableUser}
          isGuest={isGuest}
          isConnected={isConnected}
          showExecutionDetails={showExecutionDetails}
          onToggleExecutionDetails={toggleExecutionDetails}
          onShowProfile={() => setShowProfile(true)}
          onToggleSidebar={toggleSidebar}
          sidebarOpen={sidebarOpen}
        />

        {/* Content Area with Agent Flow */}
        <div className="flex-1 flex overflow-hidden">
          {/* Messages Area */}
          <motion.div 
            className="flex-1 flex flex-col overflow-hidden"
            layout
            transition={{ duration: 0.3 }}
          >
            <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4 scrollbar-hide">
              {messages.length === 0 && !currentExecution && <EmptyState />}

              <AnimatePresence mode="popLayout">
                {messages.map((message, index) => (
                  <MessageBubble
                    key={`msg-${index}`}
                    message={message}
                    messageId={index}
                    toggleStep={toggleStep}
                    expandedSteps={expandedSteps}
                    showExecutionDetails={showExecutionDetails}
                  />
                ))}
              </AnimatePresence>

              {/* Single unified execution progress box */}
              {currentExecution && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                >
                  <ExecutionProgress
                    execution={currentExecution}
                    toggleStep={toggleStep}
                    expandedSteps={expandedSteps}
                  />
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <EnhancedChatInput
              onSend={handleSend}
              onStop={handleStop}
              disabled={!isConnected}
              isProcessing={isProcessing}
              showSuggestions={messages.length === 0}
            />
          </motion.div>

          {/* Agent Flow Panel */}
          <AnimatePresence>
            {showAgentFlow && currentExecution && (
              <motion.div
                initial={{ width: 0, opacity: 0 }}
                animate={{ width: 400, opacity: 1 }}
                exit={{ width: 0, opacity: 0 }}
                transition={{ duration: 0.3, ease: 'easeInOut' }}
                className="border-l border-gray-800 bg-gray-900/50 overflow-hidden"
              >
                <div className="h-full flex flex-col">
                  <div className="p-3 border-b border-gray-800 flex items-center justify-between">
                    <h3 className="text-sm font-medium text-gray-300">Agent Flow</h3>
                    <button
                      onClick={toggleAgentFlow}
                      className="p-1 hover:bg-gray-800 rounded text-gray-400 hover:text-gray-200"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                  <div className="flex-1">
                    <AgentFlowGraph execution={currentExecution} />
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Agent Flow Toggle Button */}
        {currentExecution && !showAgentFlow && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="fixed bottom-24 right-6 p-3 bg-purple-600 hover:bg-purple-700 rounded-full shadow-lg text-white"
            onClick={toggleAgentFlow}
            title="Show Agent Flow"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </motion.button>
        )}
      </div>

      {/* Modals */}
      <LoginModal isOpen={showLogin} onClose={() => setShowLogin(false)} />
      <ProfileModal isOpen={showProfile} onClose={() => setShowProfile(false)} />
      <SettingsPanel isOpen={settingsOpen} onClose={() => toggleSettings()} />
    </div>
  );
}

export default App;
