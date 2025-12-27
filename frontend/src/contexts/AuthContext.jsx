/**
 * Auth Context - Clean implementation for fastapi-users backend.
 * Uses modern React patterns with minimal code.
 */

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

const AuthContext = createContext(null);

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Storage keys
const TOKEN_KEY = 'magentic_token';
const USER_KEY = 'magentic_user';

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isGuest, setIsGuest] = useState(false);
  const [loading, setLoading] = useState(true);

  // Clear session
  const clearSession = useCallback(() => {
    setUser(null);
    setToken(null);
    setIsAuthenticated(false);
    setIsGuest(false);
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  }, []);

  // Save session
  const saveSession = useCallback((userData, accessToken) => {
    // Add username field from email for compatibility with chat system
    const enrichedUser = {
      ...userData,
      username: userData.email?.split('@')[0] || userData.id?.toString() || 'user',
    };
    
    setUser(enrichedUser);
    setToken(accessToken);
    setIsAuthenticated(true);
    setIsGuest(userData.is_guest || false);

    // Persist non-guest sessions
    if (!userData.is_guest) {
      localStorage.setItem(USER_KEY, JSON.stringify(enrichedUser));
      localStorage.setItem(TOKEN_KEY, accessToken);
    }
  }, []);

  // Get auth header
  const getAuthHeader = useCallback(() => {
    return token ? { Authorization: `Bearer ${token}` } : {};
  }, [token]);

  // API helper with auth
  const authFetch = useCallback(async (endpoint, options = {}) => {
    const response = await fetch(`${API_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeader(),
        ...options.headers,
      },
    });
    return response;
  }, [getAuthHeader]);

  // Load saved session on mount
  useEffect(() => {
    const loadSession = async () => {
      try {
        const savedToken = localStorage.getItem(TOKEN_KEY);
        const savedUser = localStorage.getItem(USER_KEY);

        if (savedToken && savedUser) {
          const parsedUser = JSON.parse(savedUser);
          
          // Restore session immediately
          setToken(savedToken);
          setUser(parsedUser);
          setIsAuthenticated(true);
          setIsGuest(parsedUser.is_guest || false);

          // Verify token is still valid in background
          try {
            const response = await fetch(`${API_URL}/auth/me`, {
              headers: { Authorization: `Bearer ${savedToken}` },
            });
            
            if (response.ok) {
              const userData = await response.json();
              // Add username field for compatibility
              const enrichedUser = {
                ...userData,
                username: userData.email?.split('@')[0] || userData.id?.toString() || 'user',
              };
              setUser(enrichedUser);
              localStorage.setItem(USER_KEY, JSON.stringify(enrichedUser));
            } else if (response.status === 401) {
              // Token expired, clear session
              clearSession();
            }
          } catch (err) {
            console.warn('Background auth check failed:', err);
          }
        }
      } catch (error) {
        console.error('Failed to load session:', error);
        clearSession();
      } finally {
        setLoading(false);
      }
    };

    loadSession();
  }, [clearSession]);

  // ============== Auth Methods ==============

  /**
   * Login with email and password.
   * fastapi-users uses OAuth2 form data format.
   */
  const login = async (email, password) => {
    try {
      // fastapi-users expects form data for login
      const formData = new URLSearchParams();
      formData.append('username', email); // fastapi-users uses 'username' field for email
      formData.append('password', password);

      const response = await fetch(`${API_URL}/auth/jwt/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Login failed');
      }

      const data = await response.json();
      
      // Get user profile
      const profileResponse = await fetch(`${API_URL}/auth/me`, {
        headers: { Authorization: `Bearer ${data.access_token}` },
      });
      
      if (!profileResponse.ok) {
        throw new Error('Failed to get user profile');
      }
      
      const userData = await profileResponse.json();
      saveSession(userData, data.access_token);
      
      return { success: true, user: userData };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  /**
   * Register new user.
   */
  const register = async (email, password, displayName = null) => {
    try {
      const response = await fetch(`${API_URL}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email,
          password,
          display_name: displayName || email.split('@')[0],
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Registration failed');
      }

      // Auto-login after registration
      return await login(email, password);
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  /**
   * Login as guest.
   */
  const loginAsGuest = async () => {
    try {
      const response = await fetch(`${API_URL}/auth/guest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) {
        throw new Error('Guest login failed');
      }

      const data = await response.json();
      saveSession(data.user, data.access_token);
      
      return { success: true, user: data.user };
    } catch (error) {
      // Fallback to local guest if API fails
      const guestUser = {
        id: 0,
        email: `guest_${Date.now()}@local`,
        display_name: 'Guest User',
        avatar_emoji: '👻',
        is_guest: true,
      };
      setUser(guestUser);
      setIsAuthenticated(true);
      setIsGuest(true);
      return { success: true, user: guestUser };
    }
  };

  /**
   * Logout.
   */
  const logout = async () => {
    try {
      if (token) {
        await authFetch('/auth/jwt/logout', { method: 'POST' });
      }
    } catch (error) {
      console.warn('Logout API call failed:', error);
    } finally {
      clearSession();
    }
  };

  /**
   * Get current user profile.
   */
  const fetchProfile = async () => {
    if (!token) return null;

    try {
      const response = await authFetch('/auth/me');
      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
        if (!isGuest) {
          localStorage.setItem(USER_KEY, JSON.stringify(userData));
        }
        return userData;
      }
      return null;
    } catch (error) {
      console.error('Failed to fetch profile:', error);
      return null;
    }
  };

  /**
   * Update user profile.
   */
  const updateProfile = async (updates) => {
    if (!token) return { success: false, error: 'Not authenticated' };

    try {
      const response = await authFetch('/auth/users/me', {
        method: 'PATCH',
        body: JSON.stringify(updates),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Update failed');
      }

      const userData = await response.json();
      setUser(userData);
      if (!isGuest) {
        localStorage.setItem(USER_KEY, JSON.stringify(userData));
      }
      
      return { success: true, user: userData };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  /**
   * Request password reset.
   */
  const forgotPassword = async (email) => {
    try {
      const response = await fetch(`${API_URL}/auth/forgot-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to send reset email');
      }

      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  /**
   * Reset password with token.
   */
  const resetPassword = async (token, newPassword) => {
    try {
      const response = await fetch(`${API_URL}/auth/reset-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, password: newPassword }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to reset password');
      }

      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  // Context value
  const value = {
    // State
    user,
    token,
    isAuthenticated,
    isGuest,
    loading,
    
    // Methods
    login,
    register,
    loginAsGuest,
    logout,
    fetchProfile,
    updateProfile,
    forgotPassword,
    resetPassword,
    getAuthHeader,
    authFetch,
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

export default AuthContext;
