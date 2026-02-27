import React, { useState } from 'react';
import { api } from '@/api';
import { useWebSocket } from '../context/WebSocketContext';

export const Settings = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [msg, setMsg] = useState<{ type: 'success' | 'error', text: string } | null>(null);
    const { isConnected } = useWebSocket();

    const handleSave = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setMsg(null);

        try {
            await api.post('/system/config/email', { username, password });
            setMsg({ type: 'success', text: '✅ Email settings updated successfully!' });
            setUsername('');
            setPassword('');
        } catch (err: any) {
            setMsg({ type: 'error', text: err.response?.data?.detail || 'Failed to update settings' });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto p-6 space-y-8">
            <div className="space-y-2">
                <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">System Settings</h1>
                <p className="text-gray-500 dark:text-gray-400">Configure global system parameters and integrations.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Connection Status Card */}
                <div className="p-6 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700">
                    <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-white flex items-center gap-2">
                        📡 System Connectivity
                    </h2>
                    <div className="flex items-center gap-3">
                        <span className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
                        <span className="text-gray-600 dark:text-gray-300">
                            WebSocket: <b>{isConnected ? 'Connected' : 'Disconnected'}</b>
                        </span>
                    </div>
                </div>

                {/* Email Configuration Card */}
                <div className="p-6 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 md:col-span-2">
                    <div className="flex items-center gap-2 mb-4">
                        <h2 className="text-xl font-semibold text-gray-800 dark:text-white">📧 Email Notifications</h2>
                        <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">Gmail Only</span>
                    </div>

                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
                        Configure the <b>Sender Account</b> (Bot). The system will use these credentials to send alerts to users.
                        Use a <a href="https://support.google.com/accounts/answer/185833" target="_blank" rel="noreferrer" className="text-indigo-600 hover:underline">Gmail App Password</a>, not your login password.
                    </p>

                    <form onSubmit={handleSave} className="space-y-4 max-w-lg">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Gmail Address</label>
                            <input
                                type="email"
                                required
                                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 dark:bg-gray-700 dark:text-white"
                                placeholder="deepguard.bot@gmail.com"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">App Password</label>
                            <input
                                type="password"
                                required
                                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 dark:bg-gray-700 dark:text-white"
                                placeholder="xxxx xxxx xxxx xxxx"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                            />
                        </div>

                        {msg && (
                            <div className={`p-3 rounded-lg text-sm ${msg.type === 'success' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
                                {msg.text}
                            </div>
                        )}

                        <button
                            type="submit"
                            disabled={loading}
                            className={`px-6 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                            {loading ? 'Saving...' : 'Save Configuration'}
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
};
