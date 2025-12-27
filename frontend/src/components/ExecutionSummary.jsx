/**
 * ExecutionSummary component - shows completed execution summary
 */

import React from 'react';
import { CheckCircle, Coins, DollarSign } from 'lucide-react';
import AgentStep from './AgentStep';

function ExecutionSummary({ execution, messageId = 'current', toggleStep, expandedSteps }) {
  if (!execution?.plan) {
    return null;
  }

  // Get token usage from execution
  const tokenUsage = execution.token_usage;
  const hasTokens = tokenUsage?.total?.total_tokens > 0;
  const hasCost = tokenUsage?.total?.total_cost > 0;
  const costFormatted = tokenUsage?.total?.cost_formatted || '$0.00';

  return (
    <div className="bg-white/30 dark:bg-gray-800/30 border border-green-500/30 rounded-lg p-3 max-w-4xl">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <CheckCircle className="w-5 h-5 text-green-500 dark:text-green-400" />
          <div>
            <h3 className="font-semibold text-green-600 dark:text-green-400">Execution Complete</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {execution.plan.total_agents} agents • {execution.plan.total_layers} layers
            </p>
          </div>
        </div>
        
        {/* Token and cost badges */}
        <div className="flex items-center gap-2">
          {hasTokens && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 bg-amber-500/10 border border-amber-500/30 rounded-full">
              <Coins className="w-3.5 h-3.5 text-amber-500" />
              <span className="text-xs font-medium text-amber-600 dark:text-amber-400">
                {tokenUsage.total.total_tokens.toLocaleString()}
              </span>
            </div>
          )}
          {(hasTokens || hasCost) && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 bg-green-500/10 border border-green-500/30 rounded-full">
              <DollarSign className="w-3.5 h-3.5 text-green-500" />
              <span className="text-xs font-medium text-green-600 dark:text-green-400">
                {costFormatted}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Token breakdown - only show if we have tokens */}
      {hasTokens && (
        <div className="mb-4 p-2 bg-gray-50/50 dark:bg-gray-800/50 rounded-lg border border-gray-200/50 dark:border-gray-700/50">
          <div className="grid grid-cols-4 gap-4 text-center text-xs">
            <div>
              <p className="text-gray-500 dark:text-gray-400">Planning</p>
              <p className="font-medium text-gray-700 dark:text-gray-300">
                {tokenUsage.planning?.total_tokens?.toLocaleString() || 0}
              </p>
            </div>
            <div>
              <p className="text-gray-500 dark:text-gray-400">Input</p>
              <p className="font-medium text-gray-700 dark:text-gray-300">
                {tokenUsage.total.prompt_tokens?.toLocaleString() || 0}
              </p>
            </div>
            <div>
              <p className="text-gray-500 dark:text-gray-400">Output</p>
              <p className="font-medium text-gray-700 dark:text-gray-300">
                {tokenUsage.total.completion_tokens?.toLocaleString() || 0}
              </p>
            </div>
            <div>
              <p className="text-gray-500 dark:text-gray-400">Cost</p>
              <p className="font-medium text-green-600 dark:text-green-400">
                {costFormatted}
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="space-y-1">
        {execution.plan.agents.map((agent, idx) => {
          const agentStatus = execution.agents?.find(
            (a) => a.agent_id === agent.agent_id
          );
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

export default ExecutionSummary;
