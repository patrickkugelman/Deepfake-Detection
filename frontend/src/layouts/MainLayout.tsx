import { ReactNode } from 'react';
import { SidebarProvider } from '@/components/ui/sidebar';
import { ChatAssistant } from "@/components/shared/ChatAssistant";
import { AppSidebar } from "@/components/shared/AppSidebar";
import { Navbar } from "@/components/shared/Navbar";
import { useAnalysisBanner } from "@/hooks/useAnalysisBanner";
import { cn } from "@/lib/utils";

interface MainLayoutProps {
  children: ReactNode;
}

/**
 * MainLayout - Layout-ul principal al aplicației DeepGuard.
 * Include Sidebar, Navbar și Asistentul AI (Cortex Chat).
 */
export const MainLayout = ({ children }: MainLayoutProps) => {
  // Reserve space so the globally-rendered <AnalysisBanner /> (App.tsx) never covers the Navbar
  const { variant } = useAnalysisBanner();

  return (
    <SidebarProvider>
      <div className={cn(
        "min-h-screen flex w-full bg-background flex-col overflow-hidden",
        variant !== null && "pt-10"
      )}>
        {/* Top Navbar - Fixat în partea de sus */}
        <Navbar />

        <div className="flex flex-1 overflow-hidden relative">
          {/* Bara Laterală de Navigare */}
          <AppSidebar />

          {/* Zona Principală de Conținut */}
          <main className="flex-1 overflow-y-auto bg-slate-950/20">
            <div className="animate-in fade-in duration-500">
              {children}
            </div>
          </main>

          {/* Asistentul de Chat (Plutește în dreapta) */}
          <ChatAssistant />
        </div>
      </div>
    </SidebarProvider>
  );
};

// Păstrăm și exportul default pentru flexibilitate maximă
export default MainLayout;