# Changelog

All notable changes to Magentic will be documented in this file.

## [Latest] - 2024-12-25

### Added
- **Layer Barrier Synchronization**: Implemented barrier nodes between execution layers to ensure all agents in a layer complete before the next layer starts, preventing race conditions
- **Conversation History Tracking**: Agents now receive context from previous conversation steps, enabling better continuity across the execution flow
- **Debug State Visualization**: Added `DEBUG_STATE` environment variable to enable real-time state visualization showing agent outputs, lengths, and execution flow
- **UI Display Limit Configuration**: Added `UI_DISPLAY_LIMIT` setting to control character limit for agent output display (default 200, minimum 50)
- **Enhanced JSON Parsing**: Improved meta-coordinator to handle markdown-wrapped JSON responses and clean them before parsing
- **Frontend State Management**: Fixed execution state clearing between queries to prevent UI inconsistencies

### Fixed
- **Empty Context Bug**: Fixed critical issue where agent outputs were being stored with leading newlines, causing context parsing to fail when split on `\n\n`. Now outputs are stripped before building context strings
- **Last Agent Loading State**: Fixed frontend issue where the last agent would remain in "loading" state on subsequent queries due to stale execution state
- **Agent Output Retrieval**: Ensured proper state merging with LangGraph's `merge_dicts` reducer for `agent_outputs` dictionary
- **WebSocket Event Timing**: Improved event ordering to ensure `agent_complete` events are fully processed before the final `complete` event
- **JSON Response Handling**: Added pre-parsing cleanup to remove markdown code blocks from LLM responses, reducing fallback warnings

### Changed
- **Improved Coordinator Prompts**: Enhanced meta-coordinator system prompt with explicit JSON structure definition and clearer examples
- **Output Formatting**: Agent outputs are now stripped of leading/trailing whitespace to prevent parsing issues
- **State Visualization**: Made state snapshots optional via environment variable instead of always-on debug prints
- **Logging Improvements**: Added comprehensive logging for state flow tracking (STORING, RETRIEVED, BARRIER) during execution

### Technical Details
- **Layer Barrier Implementation**: 
  - Added `_create_layer_barrier()` method to create barrier nodes that check all agents in a layer have outputs
  - Modified `_add_dynamic_edges()` to use layer barriers: START → Layer 0 → Barrier 0 → Layer 1 → Barrier 1 → END
  - Barriers return state unchanged, just synchronizing execution flow

- **Conversation History Schema**:
  - Added `conversation_history: Annotated[List[Dict[str, str]], operator.add]` to MagenticState
  - Each entry contains: agent_id, role, task, input_context, output, layer, timestamp
  - Last 3 steps shown to agents for context awareness

- **State Visualization Features**:
  - Unicode table showing all agent_outputs with ID, length, and 60-char preview
  - State info display: query, current layer, total layers, conversation steps
  - BEFORE/AFTER snapshots for each agent execution
  - Python 3.13 compatible (no rich library dependency)

## Configuration

### New Environment Variables

```bash
# Enable state visualization for debugging
DEBUG_STATE=true

# Set character limit for UI display (default 200, min 50)
UI_DISPLAY_LIMIT=200
```

### Updated Configuration
- Agent prompts now include output length instruction: "Keep your response under X characters"
- Meta-coordinator prompt enhanced with explicit JSON schema and examples

## Migration Notes

No database migrations required for this update. All changes are backward compatible with existing installations.

## Known Issues

None currently identified.

## Contributors

- Enhanced execution reliability with layer barriers
- Improved context passing between agents
- Better error handling and debugging capabilities
- Cleaner JSON parsing with fallback mechanisms
