import { Badge, Separator, SidebarTrigger, Tooltip, TooltipContent, TooltipTrigger } from '@databricks/appkit-ui/react';
import { Info } from 'lucide-react';
import { AgentSwitcher } from '../components/AgentSwitcher';
import { CommandMenu } from './CommandMenu';

// The persistent top bar inside the content inset: the sidebar toggle, the
// agent/experiment "project switcher", and global actions (command palette). It also
// carries the app's load-bearing honesty note (mechanically-derived metrics; $ =
// estimate) so that trust framing stays visible on every route. TooltipProvider is
// supplied by SidebarProvider, so Tooltip works here without a local provider.
export function TopBar() {
  return (
    <header className="sticky top-0 z-20 flex h-14 shrink-0 items-center gap-2 border-b bg-background/95 px-4 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <SidebarTrigger className="-ml-1" />
      <Separator orientation="vertical" className="mr-1 h-5" />
      <AgentSwitcher />

      <div className="ml-auto flex items-center gap-2">
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="hidden cursor-default items-center gap-1.5 text-xs text-muted-foreground lg:flex">
              <Info className="h-3.5 w-3.5" />
              Mechanically derived
              <Badge variant="outline">$ = estimate</Badge>
            </span>
          </TooltipTrigger>
          <TooltipContent className="max-w-xs">
            Every metric is mechanically derived from trace metadata (tokens, timestamps, tool spans) — no model in the
            loop. Dollar figures are estimates. One MLflow experiment per agent.
          </TooltipContent>
        </Tooltip>
        <CommandMenu />
      </div>
    </header>
  );
}
