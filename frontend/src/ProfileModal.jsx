import React, { useState, useEffect } from 'react';
import { User, Settings, X, Save, History, LogOut } from 'lucide-react';
import { useAuth } from './contexts/AuthContext';

function ProfileModal({ isOpen, onClose }) {
  const { user, isGuest, logout } = useAuth();
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [editing, setEditing] = useState(false);
  const [formData, setFormData] = useState({
    display_name: '',
    avatar_emoji: '👤'
  });

  useEffect(() => {
    if (isOpen && user && !isGuest) {
      fetchProfile();
    } else if (isOpen && isGuest) {
      // For guests, use user data directly
      setProfile({
        ...user,
        stats: { total_queries: 0, total_agents_executed: 0 }
      });
      setFormData({
        display_name: user.display_name || '',
        avatar_emoji: user.avatar_emoji || '👤'
      });
    }
  }, [isOpen, user, isGuest]);

  const fetchProfile = async () => {
    if (!user || isGuest) return;
    
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/profile/${user.username}`);
      const data = await response.json();
      setProfile(data);
      setFormData({
        display_name: data.display_name || '',
        avatar_emoji: data.avatar_emoji || '👤'
      });
    } catch (error) {
      console.error('Failed to fetch profile:', error);
    }
    setLoading(false);
  };

  const saveProfile = async () => {
    if (isGuest) return; // Guests can't save profiles
    
    try {
      await fetch(`http://localhost:8000/profile/${user.username}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      setEditing(false);
      fetchProfile();
    } catch (error) {
      console.error('Failed to save profile:', error);
    }
  };

  const handleLogout = () => {
    logout();
    onClose();
  };

  const emojis = ['👤', '🧑', '👨', '👩', '🦸', '🦹', '👽', '🤖', '🐱', '🐶', '🦊', '🐼'];

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-gray-900 border border-purple-500/30 rounded-lg p-6 w-full max-w-md">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-bold text-purple-300">Profile</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-200">
            <X className="w-5 h-5" />
          </button>
        </div>

        {loading ? (
          <div className="text-center text-gray-400">Loading...</div>
        ) : profile ? (
          <div className="space-y-4">
            {/* Guest Warning */}
            {isGuest && (
              <div className="p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                <p className="text-sm text-yellow-400 text-center">
                  ⚠️ Guest mode - Conversations are NOT saved. Register to keep your history!
                </p>
              </div>
            )}

            {/* Avatar and Display Name */}
            <div className="flex items-center gap-4">
              <div className="text-5xl">{formData.avatar_emoji}</div>
              <div className="flex-1">
                {editing ? (
                  <input
                    type="text"
                    value={formData.display_name}
                    onChange={(e) => setFormData({...formData, display_name: e.target.value})}
                    className="w-full bg-gray-800 border border-purple-500/30 rounded px-3 py-2 text-gray-100"
                    placeholder="Display Name"
                  />
                ) : (
                  <div>
                    <div className="text-lg font-semibold text-gray-100">{profile.display_name}</div>
                    <div className="text-sm text-gray-400">@{profile.username}</div>
                  </div>
                )}
              </div>
            </div>

            {/* Emoji Selector (when editing and not guest) */}
            {editing && !isGuest && (
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-sm text-gray-400 mb-2">Choose Avatar:</div>
                <div className="grid grid-cols-6 gap-2">
                  {emojis.map((emoji) => (
                    <button
                      key={emoji}
                      onClick={() => setFormData({...formData, avatar_emoji: emoji})}
                      className={`text-2xl p-2 rounded hover:bg-purple-500/20 ${
                        formData.avatar_emoji === emoji ? 'bg-purple-500/30 ring-2 ring-purple-500' : ''
                      }`}
                    >
                      {emoji}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Statistics (only for registered users) */}
            {!isGuest && profile.stats && (
              <div className="bg-gray-800/50 rounded-lg p-4 space-y-2">
                <div className="text-sm font-medium text-purple-300">Statistics</div>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-gray-400">Total Queries</div>
                    <div className="text-lg font-semibold text-gray-100">{profile.stats.total_queries || 0}</div>
                  </div>
                  <div>
                    <div className="text-gray-400">Agents Used</div>
                    <div className="text-lg font-semibold text-gray-100">{profile.stats.total_agents_executed || 0}</div>
                  </div>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-2">
              {editing ? (
                <>
                  <button
                    onClick={saveProfile}
                    className="flex-1 bg-purple-600 hover:bg-purple-500 text-white rounded-lg px-4 py-2 flex items-center justify-center gap-2"
                  >
                    <Save className="w-4 h-4" />
                    Save Changes
                  </button>
                  <button
                    onClick={() => {
                      setEditing(false);
                      setFormData({
                        display_name: profile.display_name,
                        avatar_emoji: profile.avatar_emoji
                      });
                    }}
                    className="flex-1 bg-gray-700 hover:bg-gray-600 text-white rounded-lg px-4 py-2"
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <>
                  {!isGuest && (
                    <button
                      onClick={() => setEditing(true)}
                      className="flex-1 bg-purple-600 hover:bg-purple-500 text-white rounded-lg px-4 py-2 flex items-center justify-center gap-2"
                    >
                      <Settings className="w-4 h-4" />
                      Edit Profile
                    </button>
                  )}
                  <button
                    onClick={handleLogout}
                    className="flex-1 bg-red-600 hover:bg-red-500 text-white rounded-lg px-4 py-2 flex items-center justify-center gap-2"
                  >
                    <LogOut className="w-4 h-4" />
                    {isGuest ? 'Exit Guest Mode' : 'Logout'}
                  </button>
                </>
              )}
            </div>
          </div>
        ) : (
          <div className="text-center text-gray-400">Failed to load profile</div>
        )}
      </div>
    </div>
  );
}

export default ProfileModal;
