import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import {
  Button,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from '@databricks/appkit-ui/react';
import { Search } from 'lucide-react';
import { ALL_NAV, agentSearch } from '../lib/navigation';
import { useAgent } from '../context/agent-context';

// A ⌘K / Ctrl-K command palette for jumping between sections and switching agents.
// Pure client navigation over the same IA config + agent context the rest of the
// shell uses — no data logic of its own.
export function CommandMenu() {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const { agents, selected, selectAgent } = useAgent();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.key === 'k' || e.key === 'K') && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  const search = agentSearch(selected?.agent_name);

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        className="gap-2 text-muted-foreground"
        onClick={() => setOpen(true)}
        aria-label="Open command menu"
      >
        <Search className="h-4 w-4" />
        <span className="hidden sm:inline">Search…</span>
        <kbd className="ml-1 hidden rounded border bg-muted px-1.5 font-mono text-[10px] sm:inline">⌘K</kbd>
      </Button>
      <CommandDialog
        open={open}
        onOpenChange={setOpen}
        title="Command menu"
        description="Jump to a section or switch the active agent"
      >
        <CommandInput placeholder="Jump to a section or switch agent…" />
        <CommandList>
          <CommandEmpty>No results.</CommandEmpty>
          <CommandGroup heading="Navigate">
            {ALL_NAV.map((item) => (
              <CommandItem
                key={item.key}
                value={`nav ${item.label} ${item.title}`}
                onSelect={() => {
                  void navigate({ pathname: item.path, search });
                  setOpen(false);
                }}
              >
                {item.label}
              </CommandItem>
            ))}
          </CommandGroup>
          {agents.length > 0 && (
            <>
              <CommandSeparator />
              <CommandGroup heading="Switch agent">
                {agents.map((a) => (
                  <CommandItem
                    key={a.agent_name}
                    value={`agent ${a.agent_name}`}
                    onSelect={() => {
                      selectAgent(a.agent_name);
                      setOpen(false);
                    }}
                  >
                    {a.agent_name}
                  </CommandItem>
                ))}
              </CommandGroup>
            </>
          )}
        </CommandList>
      </CommandDialog>
    </>
  );
}
