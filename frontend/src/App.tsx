import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate, useLocation, useNavigate } from "react-router-dom";
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
import { WebSocketProvider } from "./context/WebSocketContext";
import { AnalysisProvider } from "./contexts/AnalysisContext";
import { useEffect, useState, useCallback } from "react";
import { getMyActiveJobs, ActiveJob } from "@/api";
import { Loader2, X } from "lucide-react";

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

// Active Analysis Banner
function ActiveAnalysisBanner() {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const [activeJobs, setActiveJobs] = useState<ActiveJob[]>([]);
  const [dismissed, setDismissed] = useState(false);

  const fetchActive = useCallback(async () => {
    if (!isAuthenticated) return;
    try {
      const jobs = await getMyActiveJobs();
      setActiveJobs(jobs);
      if (jobs.length === 0) setDismissed(false); // reset când nu mai sunt joburi active
    } catch {
      // silently fail
    }
  }, [isAuthenticated]);

  useEffect(() => {
    if (!isAuthenticated) return;
    fetchActive();
    const interval = setInterval(fetchActive, 5000);
    return () => clearInterval(interval);
  }, [isAuthenticated, fetchActive]);

  if (!isAuthenticated || dismissed || activeJobs.length === 0) return null;

  const job = activeJobs[0];
  const progress = job.progress || 0;

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-primary text-primary-foreground px-4 py-2.5 flex items-center justify-between gap-4 shadow-lg">
      <div className="flex items-center gap-3 flex-1 min-w-0">
        <Loader2 className="h-4 w-4 animate-spin shrink-0" />
        <span className="text-sm font-medium truncate">
          ⏳ Analiză în curs: <span className="font-bold">{job.filename || 'fișier'}</span>
          {job.status_message && <span className="opacity-80 ml-2">— {job.status_message}</span>}
        </span>
        <div className="hidden sm:flex items-center gap-2 shrink-0">
          <div className="w-24 bg-primary-foreground/20 rounded-full h-1.5">
            <div className="bg-primary-foreground h-1.5 rounded-full transition-all duration-500" style={{ width: `${progress}%` }} />
          </div>
          <span className="text-xs opacity-80">{progress}%</span>
        </div>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <button
          onClick={() => navigate('/jobs')}
          className="text-xs underline underline-offset-2 opacity-90 hover:opacity-100 whitespace-nowrap"
        >
          Vezi progresul
        </button>
        <button onClick={() => setDismissed(true)} className="opacity-70 hover:opacity-100 ml-1">
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

function AppContent() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
      <ActiveAnalysisBanner />
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
            <AnalysisProvider>
              <AuthProvider>
                <AppContent />
              </AuthProvider>
            </AnalysisProvider>
          </WebSocketProvider>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
