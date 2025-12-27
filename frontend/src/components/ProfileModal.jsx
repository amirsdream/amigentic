/**
 * Profile Modal - Simplified version for fastapi-users.
 * Clean UI showing user profile with edit capability.
 * Uses plain React state (no external form library).
 */

import React, { useState, useEffect } from 'react';
import { X, Save, LogOut, Loader2, Check, Coins, DollarSign } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

const EMOJI_OPTIONS = ['👤', '🧑', '👨', '👩', '🦸', '🦹', '👽', '🤖', '🐱', '🐶', '🦊', '🐼', '🦄', '🐸', '🦋', '🌟'];
const THEME_OPTIONS = [
  { value: 'dark', label: 'Dark' },
  { value: 'light', label: 'Light' },
  { value: 'system', label: 'System' },
];

export default function ProfileModal({ isOpen, onClose }) {
  const { user, isGuest, logout, updateProfile } = useAuth();
  const [isEditing, setIsEditing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  
  // Form state
  const [formData, setFormData] = useState({
    display_name: '',
    avatar_emoji: '👤',
    theme: 'dark',
  });

  // Animation on open and load user data
  useEffect(() => {
    if (isOpen) {
      requestAnimationFrame(() => setIsVisible(true));
      if (user) {
        setFormData({
          display_name: user.display_name || '',
          avatar_emoji: user.avatar_emoji || '👤',
          theme: user.theme || 'dark',
        });
      }
    } else {
      setIsVisible(false);
      setIsEditing(false);
    }
  }, [isOpen, user]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (isGuest) return;
    
    setIsLoading(true);
    try {
      const result = await updateProfile(formData);
      if (result.success) {
        setIsSaved(true);
        setIsEditing(false);
        setTimeout(() => setIsSaved(false), 2000);
      }
    } catch (error) {
      console.error('Failed to update profile:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancel = () => {
    setIsEditing(false);
    // Reset form to user data
    if (user) {
      setFormData({
        display_name: user.display_name || '',
        avatar_emoji: user.avatar_emoji || '👤',
        theme: user.theme || 'dark',
      });
    }
  };

  const handleLogout = () => {
    logout();
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div
      className={`fixed inset-0 flex items-center justify-center z-50 transition-all duration-200 ${
        isVisible ? 'bg-black/50 backdrop-blur-sm' : 'bg-black/0'
      }`}
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        className={`bg-white dark:bg-gray-900 rounded-2xl shadow-2xl border border-slate-200 dark:border-gray-700 w-full max-w-md mx-4 overflow-hidden transition-all duration-200 ${
          isVisible ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-200 dark:border-gray-700 bg-gradient-to-r from-violet-500/10 to-purple-500/10">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-2xl shadow-lg">
              {formData.avatar_emoji}
            </div>
            <div>
              <h2 className="text-xl font-semibold text-slate-800 dark:text-white">
                {user?.display_name || user?.email?.split('@')[0] || 'User'}
              </h2>
              <p className="text-sm text-slate-500 dark:text-gray-400">
                {isGuest ? 'Guest Account' : user?.email}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-gray-800 text-slate-400 hover:text-slate-600 dark:hover:text-gray-200 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Display Name */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-gray-300 mb-2">
              Display Name
            </label>
            <input
              type="text"
              disabled={isGuest || !isEditing}
              value={formData.display_name}
              onChange={(e) => setFormData(prev => ({ ...prev, display_name: e.target.value }))}
              className="w-full px-4 py-3 rounded-xl bg-slate-50 dark:bg-gray-800 border border-slate-200 dark:border-gray-700 text-slate-800 dark:text-white disabled:opacity-60 disabled:cursor-not-allowed focus:ring-2 focus:ring-violet-500 focus:border-transparent transition-all"
              placeholder="Your display name"
            />
          </div>

          {/* Avatar Emoji */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-gray-300 mb-2">
              Avatar
            </label>
            <div className="flex flex-wrap gap-2">
              {EMOJI_OPTIONS.map((emoji) => (
                <button
                  key={emoji}
                  type="button"
                  disabled={isGuest || !isEditing}
                  onClick={() => setFormData(prev => ({ ...prev, avatar_emoji: emoji }))}
                  className={`w-10 h-10 rounded-lg text-xl flex items-center justify-center transition-all ${
                    formData.avatar_emoji === emoji
                      ? 'bg-violet-500/20 border-2 border-violet-500 scale-110'
                      : 'bg-slate-100 dark:bg-gray-800 border border-slate-200 dark:border-gray-700 hover:border-violet-500/50'
                  } ${(isGuest || !isEditing) ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer'}`}
                >
                  {emoji}
                </button>
              ))}
            </div>
          </div>

          {/* Theme */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-gray-300 mb-2">
              Theme
            </label>
            <div className="flex gap-2">
              {THEME_OPTIONS.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  disabled={isGuest || !isEditing}
                  onClick={() => setFormData(prev => ({ ...prev, theme: option.value }))}
                  className={`flex-1 px-4 py-2 rounded-xl text-center text-sm font-medium transition-all ${
                    formData.theme === option.value
                      ? 'bg-violet-500/20 border-2 border-violet-500 text-violet-600 dark:text-purple-400'
                      : 'bg-slate-100 dark:bg-gray-800 border border-slate-200 dark:border-gray-700 text-slate-600 dark:text-gray-400'
                  } ${(isGuest || !isEditing) ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer'}`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          {/* Stats */}
          {user && (
            <div className="grid grid-cols-2 gap-4 p-4 bg-slate-50 dark:bg-gray-800/50 rounded-xl">
              <div className="text-center">
                <p className="text-2xl font-bold text-violet-600 dark:text-purple-400">
                  {user.total_queries || 0}
                </p>
                <p className="text-xs text-slate-500 dark:text-gray-400">Queries</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-violet-600 dark:text-purple-400">
                  {user.total_agents_executed || 0}
                </p>
                <p className="text-xs text-slate-500 dark:text-gray-400">Agents Run</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-amber-600 dark:text-amber-400 flex items-center justify-center gap-1">
                  <Coins className="w-5 h-5" />
                  {(user.total_tokens_used || 0).toLocaleString()}
                </p>
                <p className="text-xs text-slate-500 dark:text-gray-400">Total Tokens</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-emerald-600 dark:text-emerald-400 flex items-center justify-center gap-1">
                  <DollarSign className="w-5 h-5" />
                  {(user.total_cost || 0).toFixed(4)}
                </p>
                <p className="text-xs text-slate-500 dark:text-gray-400">Total Cost</p>
              </div>
            </div>
          )}

          {/* Guest Notice */}
          {isGuest && (
            <div className="p-4 bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/30 rounded-xl">
              <p className="text-sm text-amber-700 dark:text-amber-400">
                <strong>Guest Account:</strong> Your data is temporary. Create an account to save your settings and history.
              </p>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-3">
            {!isGuest && (
              isEditing ? (
                <>
                  <button
                    type="button"
                    onClick={handleCancel}
                    className="flex-1 py-3 px-4 rounded-xl bg-slate-100 dark:bg-gray-800 text-slate-700 dark:text-gray-300 font-medium hover:bg-slate-200 dark:hover:bg-gray-700 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={isLoading}
                    className="flex-1 py-3 px-4 rounded-xl bg-gradient-to-r from-violet-500 to-purple-600 text-white font-medium shadow-lg hover:shadow-violet-500/25 disabled:opacity-50 transition-all flex items-center justify-center gap-2"
                  >
                    {isLoading ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : isSaved ? (
                      <>
                        <Check className="w-5 h-5" />
                        Saved!
                      </>
                    ) : (
                      <>
                        <Save className="w-5 h-5" />
                        Save
                      </>
                    )}
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  onClick={() => setIsEditing(true)}
                  className="flex-1 py-3 px-4 rounded-xl bg-violet-500/10 dark:bg-purple-500/10 text-violet-600 dark:text-purple-400 font-medium hover:bg-violet-500/20 dark:hover:bg-purple-500/20 border border-violet-500/20 dark:border-purple-500/30 transition-colors"
                >
                  Edit Profile
                </button>
              )
            )}
            
            <button
              type="button"
              onClick={handleLogout}
              className="py-3 px-4 rounded-xl bg-red-500/10 text-red-600 dark:text-red-400 font-medium hover:bg-red-500/20 border border-red-500/20 transition-colors flex items-center gap-2"
            >
              <LogOut className="w-5 h-5" />
              {isGuest ? 'Exit' : 'Logout'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
