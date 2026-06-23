import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'

import { apiJson, resetCsrfToken } from './apiClient'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  const refreshUser = useCallback(async () => {
    try {
      const data = await apiJson('/auth/me')
      setUser(data.user)
      return data.user
    } catch {
      setUser(null)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refreshUser()
  }, [refreshUser])

  const login = useCallback(async ({ email, password }) => {
    const data = await apiJson('/auth/login', {
      method: 'POST',
      csrf: false,
      body: JSON.stringify({ email, password }),
    })
    resetCsrfToken()
    setUser(data.user)
    return data.user
  }, [])

  const register = useCallback(async ({ name, email, password }) => {
    const data = await apiJson('/auth/register', {
      method: 'POST',
      csrf: false,
      body: JSON.stringify({ name, email, password }),
    })
    resetCsrfToken()
    setUser(data.user)
    return data.user
  }, [])

  const logout = useCallback(async () => {
    try {
      await apiJson('/auth/logout', { method: 'POST' })
    } finally {
      resetCsrfToken()
      setUser(null)
    }
  }, [])

  const value = useMemo(
    () => ({ user, loading, login, register, logout, refreshUser }),
    [user, loading, login, register, logout, refreshUser],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used inside AuthProvider')
  return context
}
