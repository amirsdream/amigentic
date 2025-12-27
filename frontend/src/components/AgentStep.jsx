/**
 * AgentStep component - displays individual agent execution step
 */

import React from 'react';
import { Loader2, ChevronDown, ChevronUp, Coins, DollarSign } from 'lucide-react';

function AgentStep({ agent, status, index, expanded, onToggle, compact = false }) {
  // Debug: log when status changes
  React.useEffect(() => {
    console.log(`AgentStep ${agent.agent_id} status changed:`, status?.status || 'no status');
  }, [agent.agent_id, status?.status]);

  const getStatusIcon = () => {
    if (!status) {
      return <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full" />;
    }
    if (status.status === 'running') {
      return <Loader2 className="w-4 h-4 text-yellow-500 dark:text-yellow-400 animate-spin" />;
    }
    if (status.status === 'complete') {
      return <div className="w-2 h-2 bg-green-500 rounded-full" />;
    }
    return <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full" />;
  };

  const duration =
    status?.endTime && status?.startTime
      ? ((status.endTime - status.startTime) / 1000).toFixed(1)
      : null;

  // Get token usage for this agent
  const tokenUsage = status?.token_usage;
  const hasTokens = tokenUsage?.total_tokens > 0;
  const hasCost = tokenUsage?.total_cost > 0;
  const costFormatted = tokenUsage?.cost_formatted || '$0.00';

  return (
    <div
      className={`border border-gray-200 dark:border-purple-500/20 rounded-lg overflow-hidden transition-all ${
        compact ? 'bg-gray-100/30 dark:bg-gray-900/30' : 'bg-gray-100/50 dark:bg-gray-900/50'
      }`}
    >
      {/* Header */}
      <div
        className="flex items-center gap-3 p-3 cursor-pointer hover:bg-gray-200/30 dark:hover:bg-gray-700/30 transition-colors"
        onClick={onToggle}
      >
        {getStatusIcon()}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-purple-600 dark:text-purple-300">{agent.role}</span>
            {duration && <span className="text-xs text-gray-500">({duration}s)</span>}
            {hasTokens && (
              <span className="flex items-center gap-1 text-xs text-amber-600 dark:text-amber-400">
                <Coins className="w-3 h-3" />
                {tokenUsage.total_tokens.toLocaleString()}
              </span>
            )}
            {hasCost && (
              <span className="flex items-center gap-0.5 text-xs text-green-600 dark:text-green-400">
                <DollarSign className="w-3 h-3" />
                {costFormatted.replace('$', '')}
              </span>
            )}
          </div>
          {!compact && <p className="text-xs text-gray-600 dark:text-gray-400 truncate">{agent.task}</p>}
        </div>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-gray-500 dark:text-gray-400" />
        ) : (
          <ChevronDown className="w-4 h-4 text-gray-500 dark:text-gray-400" />
        )}
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div className="px-3 pb-3 pt-0 space-y-2 border-t border-gray-200 dark:border-purple-500/10">
          <div className="mt-2">
            <p className="text-xs text-gray-500 font-medium mb-1">Task:</p>
            <p className="text-sm text-gray-700 dark:text-gray-300">{agent.task}</p>
          </div>

          {status?.input && (
            <div>
              <p className="text-xs text-gray-500 font-medium mb-1">
                Input (from previous agents):
              </p>
              <div className="text-xs text-gray-700 dark:text-gray-300 bg-gray-200/50 dark:bg-gray-900/50 rounded p-2 font-mono max-h-40 overflow-y-auto">
                {status.input}
              </div>
            </div>
          )}

          {status?.output && (
            <div>
              <p className="text-xs text-gray-500 font-medium mb-1">Output:</p>
              <div className="text-xs text-gray-700 dark:text-gray-300 bg-gray-200/50 dark:bg-gray-900/50 rounded p-2 font-mono max-h-40 overflow-y-auto">
                {status.output}
              </div>
              {status.output_length > 1000 && (
                <p className="text-xs text-gray-500 mt-1">
                  Showing first 1000 of {status.output_length} characters
                </p>
              )}
            </div>
          )}

          {status?.tool_calls?.length > 0 && (
            <div>
              <p className="text-xs text-gray-500 font-medium mb-1">Tools Called:</p>
              <div className="space-y-1">
                {status.tool_calls.map((tool, idx) => (
                  <div key={idx} className="text-xs bg-gray-200/50 dark:bg-gray-900/50 rounded p-2">
                    <span className="text-purple-600 dark:text-purple-400 font-medium">{tool.name || tool}</span>
                    {tool.args && (
                      <pre className="text-gray-600 dark:text-gray-400 mt-1 text-xs overflow-x-auto">
                        {JSON.stringify(tool.args, null, 2)}
                      </pre>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex flex-wrap gap-4 text-xs text-gray-500">
            <span>Layer: {agent.layer}</span>
            {status?.status && <span>Status: {status.status}</span>}
            {duration && <span>Duration: {duration}s</span>}
            {hasTokens && (
              <span className="text-amber-600 dark:text-amber-400">
                <Coins className="w-3 h-3 inline mr-1" />
                {tokenUsage.prompt_tokens?.toLocaleString() || 0} in / {tokenUsage.completion_tokens?.toLocaleString() || 0} out
              </span>
            )}
            {(hasTokens || hasCost) && (
              <span className="text-green-600 dark:text-green-400">
                <DollarSign className="w-3 h-3 inline" />
                {costFormatted.replace('$', '')}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default AgentStep;
