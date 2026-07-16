import { lazy, Suspense } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router';
import { SidebarInset, SidebarProvider } from '@databricks/appkit-ui/react';
import { AgentProvider } from './context/AgentContext';
import { AppSidebar } from './shell/AppSidebar';
import { TopBar } from './shell/TopBar';
import { DEFAULT_PATH } from './lib/navigation';
import { LiveRefreshBoundary } from './shell/LiveRefreshBoundary';

const OverviewPage = lazy(() => import('./pages/OverviewPage').then((module) => ({ default: module.OverviewPage })));
const ComparePage = lazy(() => import('./pages/ComparePage').then((module) => ({ default: module.ComparePage })));
const ApprovalsPage = lazy(() => import('./pages/ApprovalsPage').then((module) => ({ default: module.ApprovalsPage })));
const LabelingPage = lazy(() => import('./pages/LabelingPage').then((module) => ({ default: module.LabelingPage })));
const ActivityPage = lazy(() => import('./pages/ActivityPage').then((module) => ({ default: module.ActivityPage })));
const LineagePage = lazy(() => import('./pages/LineagePage').then((module) => ({ default: module.LineagePage })));
const AddAgentPage = lazy(() => import('./pages/AddAgentPage').then((module) => ({ default: module.AddAgentPage })));
const HowItWorksPage = lazy(() =>
  import('./pages/HowItWorksPage').then((module) => ({ default: module.HowItWorksPage }))
);

function PageLoading() {
  return (
    <div className="rounded-md border p-6 text-sm text-muted-foreground" role="status" aria-live="polite">
      Loading this view…
    </div>
  );
}

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
              <Suspense fallback={<PageLoading />}>
                <Routes>
                  <Route path="/" element={<Navigate to={DEFAULT_PATH} replace />} />
                  <Route
                    path="/overview"
                    element={
                      <LiveRefreshBoundary>
                        <OverviewPage />
                      </LiveRefreshBoundary>
                    }
                  />
                  <Route
                    path="/compare"
                    element={
                      <LiveRefreshBoundary>
                        <ComparePage />
                      </LiveRefreshBoundary>
                    }
                  />
                  <Route
                    path="/approvals"
                    element={
                      <LiveRefreshBoundary>
                        <ApprovalsPage />
                      </LiveRefreshBoundary>
                    }
                  />
                  <Route path="/labeling" element={<LabelingPage />} />
                  <Route
                    path="/activity"
                    element={
                      <LiveRefreshBoundary>
                        <ActivityPage />
                      </LiveRefreshBoundary>
                    }
                  />
                  <Route
                    path="/lineage"
                    element={
                      <LiveRefreshBoundary>
                        <LineagePage />
                      </LiveRefreshBoundary>
                    }
                  />
                  <Route path="/add-agent" element={<AddAgentPage />} />
                  <Route path="/how-it-works" element={<HowItWorksPage />} />
                  <Route path="*" element={<Navigate to={DEFAULT_PATH} replace />} />
                </Routes>
              </Suspense>
            </div>
          </SidebarInset>
        </SidebarProvider>
      </AgentProvider>
    </BrowserRouter>
  );
}
