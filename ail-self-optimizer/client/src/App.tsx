import { BrowserRouter, Navigate, Route, Routes } from 'react-router';
import { SidebarInset, SidebarProvider } from '@databricks/appkit-ui/react';
import { AgentProvider } from './context/AgentContext';
import { AppSidebar } from './shell/AppSidebar';
import { TopBar } from './shell/TopBar';
import { OverviewPage } from './pages/OverviewPage';
import { ComparePage } from './pages/ComparePage';
import { ApprovalsPage } from './pages/ApprovalsPage';
import { LabelingPage } from './pages/LabelingPage';
import { ActivityPage } from './pages/ActivityPage';
import { LineagePage } from './pages/LineagePage';
import { AddAgentPage } from './pages/AddAgentPage';
import { HowItWorksPage } from './pages/HowItWorksPage';
import { DEFAULT_PATH } from './lib/navigation';

// The app shell: a left Sidebar (primary IA, active state from the route), a persistent
// top bar (agent/experiment "project switcher" + global actions + honesty note), and a
// routed content area (breadcrumb + page title + the page). Client routing (react-
// router; AppKit ships no router) makes every section URL-addressable and deep-linkable;
// the selected agent rides along in ?agent= so refresh/share restores the full view.
// AppKit's server serves index.html for non-API routes, so BrowserRouter deep links work
// without any server change.
export default function App() {
  return (
    <BrowserRouter>
      <AgentProvider>
        <SidebarProvider>
          <AppSidebar />
          <SidebarInset>
            <TopBar />
            <div className="mx-auto w-full max-w-7xl px-4 py-6 md:px-8">
              <Routes>
                <Route path="/" element={<Navigate to={DEFAULT_PATH} replace />} />
                <Route path="/overview" element={<OverviewPage />} />
                <Route path="/compare" element={<ComparePage />} />
                <Route path="/approvals" element={<ApprovalsPage />} />
                <Route path="/labeling" element={<LabelingPage />} />
                <Route path="/activity" element={<ActivityPage />} />
                <Route path="/lineage" element={<LineagePage />} />
                <Route path="/add-agent" element={<AddAgentPage />} />
                <Route path="/how-it-works" element={<HowItWorksPage />} />
                <Route path="*" element={<Navigate to={DEFAULT_PATH} replace />} />
              </Routes>
            </div>
          </SidebarInset>
        </SidebarProvider>
      </AgentProvider>
    </BrowserRouter>
  );
}
