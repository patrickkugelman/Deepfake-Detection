import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate, useLocation } from "react-router-dom";
import { AuthProvider, useAuth } from "@/context/AuthContext";
import Dashboard from "./pages/Dashboard";
import Upload from "./pages/Upload";
import Results from "./pages/Results";
import Compare from "./pages/Compare";
import Login from "./pages/Login";
import NotFound from "./pages/NotFound";
import { Register } from "./pages/Register";
import { Settings } from "./pages/Settings";
import Jobs from "./pages/Jobs";
import Admin from "./pages/Admin";
import DeveloperDashboard from "./pages/DeveloperDashboard";
import DeveloperDocs from "./pages/DeveloperDocs";
import DeveloperWebhooks from "./pages/DeveloperWebhooks";
import Billing from "./pages/Billing";
import { WebSocketProvider } from "./context/WebSocketContext";
import { AnalysisBanner } from "@/components/AnalysisBanner";

const queryClient = new QueryClient();

// Protected Route Component - redirects to login with return path
function ProtectedRoute({ children }: { children: JSX.Element }) {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) return <div className="min-h-screen flex items-center justify-center">Loading...</div>;
  if (!isAuthenticated) return <Navigate to="/login" state={{ from: location }} replace />;
  return children;
}

// Admin Route Component - only for admins
function AdminRoute({ children }: { children: JSX.Element }) {
  const { isAuthenticated, isLoading, user } = useAuth();
  const location = useLocation();

  if (isLoading) return <div className="min-h-screen flex items-center justify-center">Loading...</div>;
  if (!isAuthenticated) return <Navigate to="/login" state={{ from: location }} replace />;
  if (user?.role !== 'admin') return <Navigate to="/" replace />;
  return children;
}

function AppContent() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
      <AnalysisBanner />
      <Routes>
        {/* Public Routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Protected Routes */}
        <Route path="/" element={<ProtectedRoute><Upload /></ProtectedRoute>} />
        <Route path="/upload" element={<ProtectedRoute><Upload /></ProtectedRoute>} />
        <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
        <Route path="/history" element={<Navigate to="/jobs" replace />} />
        <Route path="/results/:id" element={<ProtectedRoute><Results /></ProtectedRoute>} />
        <Route path="/compare" element={<ProtectedRoute><Compare /></ProtectedRoute>} />
        <Route path="/settings" element={<ProtectedRoute><Settings /></ProtectedRoute>} />
        <Route path="/jobs" element={<ProtectedRoute><Jobs /></ProtectedRoute>} />

        {/* Developer Portal (Module 8) */}
        <Route path="/developer" element={<ProtectedRoute><DeveloperDashboard /></ProtectedRoute>} />
        <Route path="/developer/docs" element={<ProtectedRoute><DeveloperDocs /></ProtectedRoute>} />
        <Route path="/developer/webhooks" element={<ProtectedRoute><DeveloperWebhooks /></ProtectedRoute>} />
        <Route path="/billing" element={<ProtectedRoute><Billing /></ProtectedRoute>} />

        {/* Admin Routes */}
        <Route path="/admin" element={<AdminRoute><Admin /></AdminRoute>} />

        <Route path="*" element={<NotFound />} />
      </Routes>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <WebSocketProvider>
            <AuthProvider>
              <AppContent />
            </AuthProvider>
          </WebSocketProvider>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
