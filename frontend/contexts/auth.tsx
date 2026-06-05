"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { getMe, getToken, login as apiLogin, logout as apiLogout } from "@/lib/api";
import type { User, UserRole } from "@/lib/types";

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  canWrite: boolean;
  isAdmin: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    if (!getToken()) {
      setUser(null);
      setLoading(false);
      return;
    }
    try {
      const me = await getMe();
      setUser(me);
      localStorage.setItem("aml_role", me.role);
    } catch {
      setUser(null);
      apiLogout();
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const login = useCallback(
    async (email: string, password: string) => {
      const { access_token, role } = await apiLogin(email, password);
      localStorage.setItem("aml_token", access_token);
      localStorage.setItem("aml_role", role);
      await refresh();
    },
    [refresh]
  );

  const logout = useCallback(() => {
    apiLogout();
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({
      user,
      loading,
      canWrite: user?.role === "admin" || user?.role === "analyst" || user?.role === "manager",
      isAdmin: user?.role === "admin",
      login,
      logout,
      refresh,
    }),
    [user, loading, login, logout, refresh]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}

export function roleLabel(role: UserRole) {
  return {
    admin: "Administrator",
    manager: "Manager",
    analyst: "Compliance Analyst",
    viewer: "Viewer",
  }[role];
}
