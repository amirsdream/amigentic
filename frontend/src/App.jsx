import React, { useState, useEffect, useRef } from 'react';
import { Send, Loader2, Sparkles, ChevronDown, ChevronUp, Zap, Eye, EyeOff, User, CheckCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import ProfileModal from './ProfileModal';
import LoginModal from './LoginModal';
import { useAuth } from './contexts/AuthContext';

const API_URL = 'ws://localhost:8000/ws';

function App() {
  const { user, isAuthenticated, isGuest, loading } = useAuth();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [ws, setWs] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [currentExecution, setCurrentExecution] = useState(null);
  const [expandedSteps, setExpandedSteps] = useState(new Set());
  const [showExecutionDetails, setShowExecutionDetails] = useState(true);
  const [showProfile, setShowProfile] = useState(false);
  const [showLogin, setShowLogin] = useState(false);
  const messagesEndRef = useRef(null);
  const executionRef = useRef(null);

  // Keep ref in sync with state
  useEffect(() => {
    executionRef.current = currentExecution;
  }, [currentExecution]);

  // Show login modal if not authenticated
  useEffect(() => {
    if (!loading && !isAuthenticated) {
      setShowLogin(true);
    }
  }, [loading, isAuthenticated]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isAuthenticated && user) {
      connectWebSocket();
    }
    return () => {
      if (ws) ws.close();
    };
  }, [isAuthenticated, user]);

  const connectWebSocket = () => {
    if (!user) return;
    
    const websocket = new WebSocket(`${API_URL}?username=${encodeURIComponent(user.username)}`);

    websocket.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };

    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      // Reconnect after 3 seconds
      setTimeout(connectWebSocket, 3000);
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };

    setWs(websocket);
  };

  const handleWebSocketMessage = (data) => {
    console.log('Received:', data);

    switch (data.type) {
      case 'status':
        // Initial acknowledgment
        break;

      case 'stage':
        setCurrentExecution((prev) => ({
          ...prev,
          stage: data.stage,
          stageMessage: data.message,
        }));
        break;

      case 'plan':
        // Start new execution - initialize agents from plan with 'pending' status
        console.log('Plan received with agents:', data.data.agents);
        setCurrentExecution({
          stage: 'planned',
          plan: data.data,
          agents: data.data.agents.map(agent => ({
            ...agent,
            status: 'pending', // Will be updated to 'running' then 'complete'
          })),
          stageMessage: `Executing ${data.data.total_agents} agents across ${data.data.total_layers} layers`,
        });
        break;

      case 'agent_start':
        console.log('Agent start received:', data.data.agent_id);
        setCurrentExecution((prev) => {
          if (!prev || !prev.agents) {
            console.warn('No current execution when agent_start received');
            return prev;
          }
          
          // Update the existing agent from the plan to status 'running'
          const updatedAgents = prev.agents.map((agent) => {
            if (agent.agent_id === data.data.agent_id) {
              console.log(`Setting agent ${agent.agent_id} to running`);
              return {
                ...agent,
                status: 'running',
                input: data.data.input,  // Capture what was handed to this agent
                startTime: Date.now(),
              };
            }
            return agent;
          });
          
          return {
            ...prev,
            agents: updatedAgents,
          };
        });
        break;

      case 'agent_complete':
        console.log('Agent complete received:', {
          completed_agent_id: data.data.agent_id,
          current_agents: executionRef.current?.agents?.map(a => ({ id: a.agent_id, status: a.status })),
          full_data: data.data
        });
        setCurrentExecution((prev) => {
          if (!prev || !prev.agents) {
            console.warn('No current execution or agents to update!');
            return prev;
          }
          
          // Check if agent exists before updating
          const agentExists = prev.agents.some(a => a.agent_id === data.data.agent_id);
          if (!agentExists) {
            console.error(`Agent ${data.data.agent_id} not found in agents list!`, prev.agents.map(a => a.agent_id));
          }
          
          const updatedAgents = prev.agents.map((agent) => {
            if (agent.agent_id === data.data.agent_id) {
              console.log(`Updating agent ${agent.agent_id} from ${agent.status} to complete`);
              return {
                ...agent,
                status: 'complete',
                input: data.data.input,  // Update input in case it wasn't set on agent_start
                output: data.data.output,
                output_length: data.data.output_length,
                tool_calls: data.data.tool_calls,
                endTime: Date.now(),
              };
            }
            return agent;
          });
          
          console.log('Updated agents:', updatedAgents.map(a => ({ id: a.agent_id, status: a.status })));
          
          return {
            ...prev,
            agents: updatedAgents,
          };
        });
        break;

      case 'complete':
        // Create a deep copy to preserve execution data
        const executionData = JSON.parse(JSON.stringify(executionRef.current));
        console.log('Saving message with execution:', executionData);
        
        // Add the message WITH execution at the top
        setMessages((msgs) => [
          ...msgs,
          {
            type: 'assistant',
            content: data.data.output,
            execution: executionData,
            timestamp: new Date(),
          },
        ]);
        
        // Clear the current execution since it's now in the message
        setCurrentExecution(null);
        break;

      case 'error':
        setMessages((prev) => [
          ...prev,
          {
            type: 'error',
            content: data.message,
            timestamp: new Date(),
          },
        ]);
        setCurrentExecution(null);
        break;
    }
  };

  const handleSend = () => {
    if (!input.trim() || !ws || !isConnected) return;

    // Add user message
    setMessages((prev) => [
      ...prev,
      {
        type: 'user',
        content: input,
        timestamp: new Date(),
      },
    ]);

    // Clear previous execution state before sending new query
    setCurrentExecution(null);

    // Send to WebSocket
    ws.send(JSON.stringify({ query: input }));

    setInput('');
  };

  const toggleStep = (key) => {
    setExpandedSteps((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(key)) {
        newSet.delete(key);
      } else {
        newSet.add(key);
      }
      return newSet;
    });
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-gray-950 via-purple-950/20 to-gray-950">
      {/* Header */}
      <header className="bg-gray-900/80 backdrop-blur-lg border-b border-purple-500/20 px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Sparkles className="w-8 h-8 text-purple-400" />
            <div className="absolute inset-0 blur-xl bg-purple-500/30 rounded-full"></div>
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              Magentic
            </h1>
            <p className="text-xs text-gray-400">Magnetic Agent Networks</p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={() => setShowProfile(true)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 transition-colors"
              title="Profile"
            >
              <span className="text-lg">{user?.avatar_emoji || '👤'}</span>
              <div className="flex flex-col items-start">
                <span className="text-xs text-gray-300">{user?.display_name || user?.username}</span>
                {isGuest && <span className="text-xs text-yellow-400">Guest</span>}
              </div>
            </button>
            <button
              onClick={() => {
                console.log('Toggling execution details from', showExecutionDetails, 'to', !showExecutionDetails);
                setShowExecutionDetails(!showExecutionDetails);
              }}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 transition-colors"
              title={showExecutionDetails ? 'Hide agent details' : 'Show agent details'}
            >
              {showExecutionDetails ? (
                <Eye className="w-4 h-4 text-purple-400" />
              ) : (
                <EyeOff className="w-4 h-4 text-gray-500" />
              )}
              <span className="text-xs text-gray-400">
                Agent Details
              </span>
            </button>
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-500' : 'bg-red-500'
              } animate-pulse`}
            ></div>
            <span className="text-sm text-gray-400">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </header>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4 scrollbar-hide">
        {messages.length === 0 && !currentExecution && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Sparkles className="w-16 h-16 text-purple-400/50 mb-4" />
            <h2 className="text-xl font-semibold text-gray-300 mb-2">
              Welcome to Magentic
            </h2>
            <p className="text-gray-500 max-w-md">
              Ask me anything. I'll create a dynamic network of AI agents to solve your query.
            </p>
          </div>
        )}

        {messages.map((message, index) => (
          <Message key={index} message={message} messageId={index} toggleStep={toggleStep} expandedSteps={expandedSteps} showExecutionDetails={showExecutionDetails} />
        ))}

        {currentExecution && (
          <ExecutionProgress execution={currentExecution} toggleStep={toggleStep} expandedSteps={expandedSteps} />
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="bg-gray-900/80 backdrop-blur-lg border-t border-purple-500/20 px-6 py-4">
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask me anything..."
            className="flex-1 bg-gray-800/50 border border-purple-500/30 rounded-lg px-4 py-3 text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
            disabled={!isConnected}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || !isConnected}
            className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:from-gray-700 disabled:to-gray-700 text-white rounded-lg px-6 py-3 font-medium flex items-center gap-2 transition-all disabled:cursor-not-allowed"
          >
            <Send className="w-5 h-5" />
            Send
          </button>
        </div>
      </div>

      {/* Login Modal */}
      <LoginModal 
        isOpen={showLogin} 
        onClose={() => setShowLogin(false)}
      />

      {/* Profile Modal */}
      <ProfileModal 
        isOpen={showProfile} 
        onClose={() => setShowProfile(false)}
      />
    </div>
  );
}

function Message({ message, messageId, toggleStep, expandedSteps, showExecutionDetails }) {
  console.log('Message rendering, showExecutionDetails:', showExecutionDetails, 'has execution:', !!message.execution);
  
  if (message.type === 'user') {
    return (
      <div className="flex justify-end">
        <div className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg px-4 py-3 max-w-2xl">
          <p className="text-white">{message.content}</p>
        </div>
      </div>
    );
  }

  if (message.type === 'error') {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3">
        <p className="text-red-400">{message.content}</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {/* Show execution summary ABOVE the message */}
      {message.execution && showExecutionDetails && (
        <ExecutionSummary execution={message.execution} messageId={messageId} toggleStep={toggleStep} expandedSteps={expandedSteps} />
      )}
      
      {/* Assistant response */}
      <div className="bg-gray-800/50 border border-purple-500/20 rounded-lg px-4 py-3 max-w-4xl">
        <div className="markdown-content">
          <ReactMarkdown 
            remarkPlugins={[remarkGfm]} 
            rehypePlugins={[rehypeHighlight]}
          >
            {message.content}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}

function ExecutionProgress({ execution, toggleStep, expandedSteps }) {
  // Check if all agents are complete or if execution stage is 'complete'
  const allComplete = (execution.stage === 'complete') || 
    (execution.plan && execution.agents && 
    execution.plan.agents.length > 0 && 
    execution.agents.length === execution.plan.agents.length &&
    execution.agents.every(a => a.status === 'complete'));

  return (
    <div className="bg-gray-800/50 border border-purple-500/30 rounded-lg p-4 max-w-4xl">
      {!allComplete && (
        <div className="flex items-center gap-3 mb-4">
          <div className="relative w-4 h-4">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse-red absolute top-1 left-1"></div>
            <div className="absolute inset-0 border-2 border-red-500 rounded-full animate-spin-slow"></div>
          </div>
          <div>
            <h3 className="font-semibold text-purple-300">Processing Query</h3>
            <p className="text-sm text-gray-400">{execution.stageMessage}</p>
          </div>
        </div>
      )}

      {allComplete && (
        <div className="flex items-center gap-3 mb-4">
          <CheckCircle className="w-5 h-5 text-green-400" />
          <div>
            <h3 className="font-semibold text-green-400">Execution Complete</h3>
            <p className="text-sm text-gray-400">All agents finished successfully</p>
          </div>
        </div>
      )}

      {execution.plan && (
        <div className="space-y-2">
          <div className="text-sm text-gray-400 mb-2">
            {execution.plan.total_agents} agents • {execution.plan.total_layers} layers
          </div>
          
          {execution.plan.agents.map((agent, idx) => {
            const agentStatus = execution.agents?.find(a => a.agent_id === agent.agent_id);
            const stepKey = `current-${idx}`;
            console.log('Finding status for agent:', {
              plan_agent_id: agent.agent_id,
              execution_agents: execution.agents?.map(a => a.agent_id),
              found_status: agentStatus
            });
            return (
              <AgentStep
                key={agent.agent_id || idx}
                agent={agent}
                status={agentStatus}
                index={idx}
                expanded={expandedSteps.has(stepKey)}
                onToggle={() => toggleStep(stepKey)}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

function ExecutionSummary({ execution, messageId = 'current', toggleStep, expandedSteps, alwaysShow = false }) {
  console.log('ExecutionSummary rendering with:', execution);
  
  if (!execution || !execution.plan) {
    console.log('No execution or plan data');
    return null;
  }
  
  // Check if all agents completed
  const allComplete = execution.agents && 
    execution.plan.agents.length > 0 && 
    execution.agents.length === execution.plan.agents.length &&
    execution.agents.every(a => a.status === 'complete');
  
  return (
    <div className="bg-gray-800/30 border border-green-500/30 rounded-lg p-3 max-w-4xl">
      <div className="flex items-center gap-3 mb-4">
        <CheckCircle className="w-5 h-5 text-green-400" />
        <div>
          <h3 className="font-semibold text-green-400">Execution Complete</h3>
          <p className="text-sm text-gray-400">{execution.plan.total_agents} agents • {execution.plan.total_layers} layers</p>
        </div>
      </div>
      
      <div className="space-y-1">
        {execution.plan.agents.map((agent, idx) => {
          const agentStatus = execution.agents?.find(a => a.agent_id === agent.agent_id);
          const stepKey = `${messageId}-${idx}`;
          return (
              <AgentStep
                key={agent.agent_id || idx}
                agent={agent}
                status={agentStatus}
                index={idx}
                expanded={expandedSteps.has(stepKey)}
                onToggle={() => toggleStep(stepKey)}
                compact
              />
            );
          })}
        </div>
      </div>
    );
  }

function AgentStep({ agent, status, index, expanded, onToggle, compact = false }) {
  // Debug logging
  console.log('AgentStep render:', { 
    agent_id: agent.agent_id, 
    role: agent.role,
    status: status,
    has_status: !!status,
    status_value: status?.status 
  });

  const getStatusIcon = () => {
    if (!status) return <div className="w-2 h-2 bg-gray-500 rounded-full"></div>;
    if (status.status === 'running')
      return <Loader2 className="w-4 h-4 text-yellow-400 animate-spin" />;
    if (status.status === 'complete')
      return <div className="w-2 h-2 bg-green-500 rounded-full"></div>;
    return <div className="w-2 h-2 bg-gray-500 rounded-full"></div>;
  };

  const duration = status?.endTime && status?.startTime
    ? ((status.endTime - status.startTime) / 1000).toFixed(1)
    : null;

  return (
    <div
      className={`border border-purple-500/20 rounded-lg overflow-hidden transition-all ${
        compact ? 'bg-gray-900/30' : 'bg-gray-900/50'
      }`}
    >
      <div
        className="flex items-center gap-3 p-3 cursor-pointer hover:bg-gray-700/30 transition-colors"
        onClick={onToggle}
      >
        {getStatusIcon()}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-purple-300">{agent.role}</span>
            {duration && (
              <span className="text-xs text-gray-500">({duration}s)</span>
            )}
          </div>
          {!compact && (
            <p className="text-xs text-gray-400 truncate">{agent.task}</p>
          )}
        </div>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-gray-400" />
        ) : (
          <ChevronDown className="w-4 h-4 text-gray-400" />
        )}
      </div>
      
      {expanded && (
        <div className="px-3 pb-3 pt-0 space-y-2 border-t border-purple-500/10">
          <div className="mt-2">
            <p className="text-xs text-gray-500 font-medium mb-1">Task:</p>
            <p className="text-sm text-gray-300">{agent.task}</p>
          </div>
          
          {status?.input && (
            <div>
              <p className="text-xs text-gray-500 font-medium mb-1">Input (from previous agents):</p>
              <div className="text-xs text-gray-300 bg-gray-900/50 rounded p-2 font-mono max-h-40 overflow-y-auto">
                {status.input}
              </div>
            </div>
          )}
          
          {status?.output && (
            <div>
              <p className="text-xs text-gray-500 font-medium mb-1">Output:</p>
              <div className="text-xs text-gray-300 bg-gray-900/50 rounded p-2 font-mono max-h-40 overflow-y-auto">
                {status.output}
              </div>
              {status.output_length > 1000 && (
                <p className="text-xs text-gray-500 mt-1">Showing first 1000 of {status.output_length} characters</p>
              )}
            </div>
          )}
          
          {status?.tool_calls && status.tool_calls.length > 0 && (
            <div>
              <p className="text-xs text-gray-500 font-medium mb-1">Tools Called:</p>
              <div className="space-y-1">
                {status.tool_calls.map((tool, idx) => (
                  <div key={idx} className="text-xs bg-gray-900/50 rounded p-2">
                    <span className="text-purple-400 font-medium">{tool.name || tool}</span>
                    {tool.args && (
                      <pre className="text-gray-400 mt-1 text-xs overflow-x-auto">
                        {JSON.stringify(tool.args, null, 2)}
                      </pre>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
          
          <div className="flex gap-4 text-xs text-gray-500">
            <span>Layer: {agent.layer}</span>
            {status?.status && <span>Status: {status.status}</span>}
            {duration && <span>Duration: {duration}s</span>}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
