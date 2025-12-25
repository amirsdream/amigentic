import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isGuest, setIsGuest] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check for saved session on mount
    const savedUser = localStorage.getItem('magentic_user');
    if (savedUser) {
      try {
        const userData = JSON.parse(savedUser);
        setUser(userData);
        setIsAuthenticated(true);
        setIsGuest(userData.is_guest || false);
      } catch (error) {
        console.error('Failed to parse saved user:', error);
        localStorage.removeItem('magentic_user');
      }
    }
    setLoading(false);
  }, []);

  const login = async (username, password) => {
    try {
      const response = await fetch('http://localhost:8000/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Login failed');
      }

      const data = await response.json();
      const userData = {
        ...data.user,
        is_guest: false
      };

      setUser(userData);
      setIsAuthenticated(true);
      setIsGuest(false);
      localStorage.setItem('magentic_user', JSON.stringify(userData));

      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  const register = async (username, password) => {
    try {
      const response = await fetch('http://localhost:8000/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Registration failed');
      }

      const data = await response.json();
      const userData = {
        ...data.user,
        is_guest: false
      };

      setUser(userData);
      setIsAuthenticated(true);
      setIsGuest(false);
      localStorage.setItem('magentic_user', JSON.stringify(userData));

      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  const loginAsGuest = () => {
    const guestUser = {
      username: `guest_${Date.now()}`,
      display_name: 'Guest User',
      avatar_emoji: '👤',
      is_guest: true
    };

    setUser(guestUser);
    setIsAuthenticated(true);
    setIsGuest(true);
    // Don't save guest sessions to localStorage
    // They're ephemeral by design

    return { success: true };
  };

  const logout = () => {
    setUser(null);
    setIsAuthenticated(false);
    setIsGuest(false);
    localStorage.removeItem('magentic_user');
  };

  const value = {
    user,
    isAuthenticated,
    isGuest,
    loading,
    login,
    register,
    loginAsGuest,
    logout
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
