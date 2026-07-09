import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import type { AxiosError } from 'axios';
import { User, getMe, login as apiLogin, register as apiRegister, AuthResponse } from '@/api';

interface AuthContextType {
    user: User | null;
    token: string | null;
    login: (u: string, p: string) => Promise<void>;
    register: (u: string, p: string, e: string) => Promise<void>;
    logout: () => void;
    isLoading: boolean;
    isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [token, setToken] = useState<string | null>(localStorage.getItem('token'));
    const [isLoading, setIsLoading] = useState<boolean>(true);

    useEffect(() => {
        const initAuth = async () => {
            if (token) {
                try {
                    const userData = await getMe(token);
                    setUser(userData);
                } catch (error) {
                    // Only an actual 401 means the token is invalid/expired — log out.
                    // A transient failure (rate limit, network blip, server hiccup) used
                    // to be treated the same way, force-logging out users with a
                    // perfectly valid token any time getMe() happened to fail once.
                    const isAuthError = (error as AxiosError & { isAuthError?: boolean })?.isAuthError
                        || (error as AxiosError)?.response?.status === 401;
                    if (isAuthError) {
                        logout();
                    } else {
                        console.warn('AuthContext: getMe() failed with a non-auth error — keeping session for retry', error);
                    }
                }
            }
            setIsLoading(false);
        };
        initAuth();
    }, [token]);

    const login = async (u: string, p: string) => {
        const response = await apiLogin(u, p);
        setToken(response.access_token);
        localStorage.setItem('token', response.access_token);
        // User will be fetched by useEffect
    };

    const register = async (u: string, p: string, e: string) => {
        const response = await apiRegister(u, p, e);
        setToken(response.access_token);
        localStorage.setItem('token', response.access_token);
    };

    const logout = () => {
        setToken(null);
        setUser(null);
        localStorage.removeItem('token');
    };

    return (
        <AuthContext.Provider value={{
            user,
            token,
            login,
            register,
            logout,
            isLoading,
            isAuthenticated: !!user
        }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
}
