/**
 * Header component - app header with user info and controls
 */

import React, { memo, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Eye, EyeOff, Menu, PanelLeftClose } from 'lucide-react';

// Memoized user button to prevent unnecessary re-renders
const UserButton = memo(function UserButton({ user, isGuest, onClick }) {
  const avatarEmoji = user?.avatar_emoji || '👤';
  
  return (
    <button
      onClick={onClick}
      className="relative w-10 h-10 flex items-center justify-center rounded-full bg-gradient-to-br from-violet-500 to-purple-600 dark:from-purple-500 dark:to-violet-600 hover:from-violet-600 hover:to-purple-700 dark:hover:from-purple-600 dark:hover:to-violet-700 text-xl shadow-lg hover:shadow-violet-500/25 transition-all duration-200 hover:scale-105 active:scale-95"
      title="Open profile"
    >
      <span className="drop-shadow-sm">{avatarEmoji}</span>
      {/* Guest indicator dot */}
      {isGuest && (
        <span className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-amber-500 border-2 border-white dark:border-gray-900 rounded-full" title="Guest" />
      )}
    </button>
  );
});

// Memoized connection status
const ConnectionStatus = memo(function ConnectionStatus({ isConnected }) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-slate-100/80 dark:bg-gray-800/50 border border-slate-200/50 dark:border-gray-700/50">
      <div className="relative">
        <div
          className={`w-2.5 h-2.5 rounded-full ${
            isConnected ? 'bg-emerald-500' : 'bg-red-500'
          }`}
        />
        {isConnected && (
          <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping opacity-75" />
        )}
      </div>
      <span className="text-xs font-medium text-slate-500 dark:text-gray-400 hidden sm:inline">
        {isConnected ? 'Connected' : 'Offline'}
      </span>
    </div>
  );
});

function Header({
  user,
  isGuest,
  isConnected,
  showExecutionDetails,
  onToggleExecutionDetails,
  onShowProfile,
  onToggleSidebar,
  sidebarOpen,
}) {
  return (
    <header className="sticky top-0 z-40 bg-white/80 dark:bg-gray-900/90 backdrop-blur-xl border-b border-slate-200/80 dark:border-purple-500/20 px-4 py-3 transition-colors duration-200">
      <div className="flex items-center justify-between gap-4">
        {/* Left section - Logo and sidebar toggle */}
        <div className="flex items-center gap-3 shrink-0">
          {/* Sidebar Toggle */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onToggleSidebar}
            className="p-2 rounded-lg hover:bg-violet-500/10 dark:hover:bg-purple-500/20 transition-colors text-slate-500 dark:text-gray-400 hover:text-violet-600 dark:hover:text-purple-400"
            title={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
          >
            {sidebarOpen ? (
              <PanelLeftClose className="w-5 h-5" />
            ) : (
              <Menu className="w-5 h-5" />
            )}
          </motion.button>

          {/* Logo */}
          <div className="relative">
            <Sparkles className="w-7 h-7 text-violet-600 dark:text-purple-400" />
            <div className="absolute inset-0 blur-xl bg-violet-500/20 dark:bg-purple-500/30 rounded-full" />
          </div>

          {/* Title */}
          <div className="hidden sm:block">
            <h1 className="text-xl font-semibold bg-gradient-to-r from-violet-600 to-fuchsia-600 dark:from-purple-400 dark:to-pink-400 bg-clip-text text-transparent">
              Magentic
            </h1>
            <p className="text-[10px] text-slate-500 dark:text-gray-500">Magnetic Agent Networks</p>
          </div>
        </div>

        {/* Right section - Controls with stable layout */}
        <div className="flex items-center gap-3">
          {/* Toggle Execution Details */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onToggleExecutionDetails}
            className="flex items-center gap-2 px-3 py-2 rounded-xl bg-slate-100/80 dark:bg-gray-800/50 hover:bg-violet-500/10 dark:hover:bg-purple-500/20 border border-slate-200/50 dark:border-gray-700/50 hover:border-violet-500/30 dark:hover:border-purple-500/30 transition-all duration-200"
            title={showExecutionDetails ? 'Hide agent details' : 'Show agent details'}
          >
            {showExecutionDetails ? (
              <Eye className="w-4 h-4 text-violet-600 dark:text-purple-400" />
            ) : (
              <EyeOff className="w-4 h-4 text-slate-400 dark:text-gray-500" />
            )}
            <span className="text-xs font-medium text-slate-500 dark:text-gray-400 hidden sm:inline">
              {showExecutionDetails ? 'Details On' : 'Details Off'}
            </span>
          </motion.button>

          {/* Connection Status */}
          <ConnectionStatus isConnected={isConnected} />

          {/* User Profile Button - circular avatar */}
          <UserButton 
            user={user} 
            isGuest={isGuest} 
            onClick={onShowProfile} 
          />
        </div>
      </div>
    </header>
  );
}

export default memo(Header);
