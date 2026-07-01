import { useCallback, useEffect, useState } from 'react';
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Checkbox,
  Input,
  Label,
  Progress,
  RadioGroup,
  RadioGroupItem,
  Separator,
  Skeleton,
} from '@databricks/appkit-ui/react';
import {
  WIZARD_STEPS,
  canAdvance,
  clampStep,
  createExperimentBody,
  creationMessage,
  dataGateView,
  freshnessMessage,
  initialWizardState,
  isLastStep,
  registerBody,
  registerMessage,
  requirementsBody,
  resolvedFromCreation,
  resolvedFromValidation,
  stepValidation,
  toggleGoal,
  validateExperimentBody,
  type CreationResponse,
  type RegisterResponse,
  type RequirementsResponse,
  type Tone,
  type ToneMessage,
  type ValidationResponse,
  type WizardState,
} from '../lib/onboarding';

const API = {
  requirements: '/api/onboarding/requirements',
  validate: '/api/onboarding/experiment/validate',
  create: '/api/onboarding/experiment/create',
  register: '/api/onboarding/register',
} as const;

const TONE_CLASS: Record<Tone, string> = {
  success: 'text-emerald-700 dark:text-emerald-300',
  warning: 'text-amber-700 dark:text-amber-300',
  error: 'text-destructive',
  info: 'text-muted-foreground',
};

// POST JSON and interpret the response fail-closed. A 401 maps to an honest
// "sign in" message; a network/parse failure is an error — never a fabricated
// success. The body identity (actor) is NEVER sent; the server resolves it.
async function postJson<T>(url: string, body: unknown): Promise<{ ok: boolean; status: number; body: T }> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const parsed = (await res.json().catch(() => ({}))) as T;
  return { ok: res.ok, status: res.status, body: parsed };
}

function Message({ message }: { message: ToneMessage | null }) {
  if (!message) return null;
  return <p className={`text-sm ${TONE_CLASS[message.tone]}`}>{message.text}</p>;
}

// The "Add an agent" wizard (docs/ONBOARDING_WIZARD.md slice 1). A stepper that
// produces ONE registered agent: validate/create a fresh experiment → choose the
// goal(s) → accept the real data gates → register (reusing ail.publish_versions,
// so the agent appears in the AgentSwitcher). Every gate/scorer/floor fact is
// fetched from the Python engine — nothing is re-derived here.
export function OnboardingWizard({
  onRegistered,
  onClose,
}: {
  onRegistered: (agentName: string) => void;
  onClose: () => void;
}) {
  const [state, setState] = useState<WizardState>(initialWizardState);
  const [requirements, setRequirements] = useState<RequirementsResponse | null>(null);
  const [reqError, setReqError] = useState<string | null>(null);
  const [registerResult, setRegisterResult] = useState<RegisterResponse | null>(null);

  const patch = useCallback((p: Partial<WizardState>) => setState((s) => ({ ...s, ...p })), []);

  // The goal catalog + the data gates for the current selection come from Python
  // (two-tier: no gate/scorer logic in TS). Refetched whenever the selection
  // changes; the empty-selection fetch on mount populates the catalog.
  useEffect(() => {
    let live = true;
    postJson<RequirementsResponse>(API.requirements, requirementsBody(state.goals))
      .then(({ ok, status, body }) => {
        if (!live) return;
        if (!ok || body.outcome === 'error') {
          setReqError(
            status === 401 ? 'Sign in to onboard an agent.' : (body.error ?? 'Could not load goal requirements.')
          );
          return;
        }
        setReqError(null);
        setRequirements(body);
      })
      .catch(() => live && setReqError('Network error loading goal requirements.'));
    return () => {
      live = false;
    };
  }, [state.goals]);

  const validation = stepValidation(state);
  const stepKey = WIZARD_STEPS[state.stepIndex].key;

  function finish() {
    setRegisterResult(null);
    if (!state.resolved) return;
    void postJson<RegisterResponse>(
      API.register,
      registerBody(state.agentName, state.resolved.experiment_id, state.goals)
    )
      .then(({ status, body }) => {
        if (status === 401) {
          setRegisterResult({ outcome: 'error', error: 'Not authenticated — sign in to register.' });
          return;
        }
        setRegisterResult(body);
        if (body.outcome === 'registered' && body.agent_name) onRegistered(body.agent_name);
      })
      .catch(() => setRegisterResult({ outcome: 'error', error: 'Network error during registration.' }));
  }

  const registered = registerResult?.outcome === 'registered';

  return (
    <Card className="shadow-sm border-primary/30">
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div>
            <CardTitle>Add an agent</CardTitle>
            <CardDescription>{WIZARD_STEPS[state.stepIndex].description}</CardDescription>
          </div>
          <Button variant="ghost" onClick={onClose} aria-label="Close wizard">
            Close
          </Button>
        </div>
        <Stepper stepIndex={state.stepIndex} />
      </CardHeader>
      <CardContent className="space-y-5">
        {stepKey === 'experiment' && <ExperimentStep state={state} patch={patch} />}
        {stepKey === 'goals' && (
          <GoalsStep state={state} patch={patch} requirements={requirements} reqError={reqError} />
        )}
        {stepKey === 'data_gate' && (
          <DataGateStep state={state} patch={patch} requirements={requirements} reqError={reqError} />
        )}
        {stepKey === 'register' && <RegisterStep state={state} patch={patch} result={registerResult} />}

        <Separator />

        <div className="flex flex-wrap items-center gap-3">
          <Button
            variant="outline"
            onClick={() => patch({ stepIndex: clampStep(state.stepIndex - 1) })}
            disabled={state.stepIndex === 0 || registered}
          >
            Back
          </Button>
          {isLastStep(state) ? (
            <Button onClick={finish} disabled={!canAdvance(state) || registered}>
              Register agent
            </Button>
          ) : (
            <Button onClick={() => patch({ stepIndex: clampStep(state.stepIndex + 1) })} disabled={!canAdvance(state)}>
              Next
            </Button>
          )}
          {registered ? (
            <Button variant="outline" onClick={onClose}>
              Done
            </Button>
          ) : (
            !validation.ok && <span className="text-sm text-muted-foreground">{validation.reason}</span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function Stepper({ stepIndex }: { stepIndex: number }) {
  const pct = ((stepIndex + 1) / WIZARD_STEPS.length) * 100;
  return (
    <div className="space-y-2 pt-2">
      <Progress value={pct} />
      <ol className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {WIZARD_STEPS.map((step, i) => (
          <li
            key={step.key}
            className={
              i === stepIndex
                ? 'font-semibold text-foreground'
                : i < stepIndex
                  ? 'text-emerald-700 dark:text-emerald-300'
                  : 'text-muted-foreground'
            }
          >
            {i + 1}. {step.title}
            {i < stepIndex ? ' ✓' : ''}
          </li>
        ))}
      </ol>
    </div>
  );
}

interface StepProps {
  state: WizardState;
  patch: (p: Partial<WizardState>) => void;
}

// Page 1 — validate or create a FRESH experiment (one agent per experiment). A
// write-path: the engine only reports fresh/created when it truly verified/created.
function ExperimentStep({ state, patch }: StepProps) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<ToneMessage | null>(null);

  async function validate() {
    setBusy(true);
    setMessage(null);
    try {
      const { status, body } = await postJson<ValidationResponse>(
        API.validate,
        validateExperimentBody(state.experimentIdInput)
      );
      if (status === 401) {
        setMessage({ tone: 'error', text: 'Sign in to validate an experiment.' });
        return;
      }
      setMessage(freshnessMessage(body));
      patch({ resolved: resolvedFromValidation(body) });
    } catch {
      setMessage({ tone: 'error', text: 'Network error validating the experiment.' });
    } finally {
      setBusy(false);
    }
  }

  async function create() {
    setBusy(true);
    setMessage(null);
    try {
      const { status, body } = await postJson<CreationResponse>(
        API.create,
        createExperimentBody(state.experimentNameInput)
      );
      if (status === 401) {
        setMessage({ tone: 'error', text: 'Sign in to create an experiment.' });
        return;
      }
      setMessage(creationMessage(body));
      patch({ resolved: resolvedFromCreation(body) });
    } catch {
      setMessage({ tone: 'error', text: 'Network error creating the experiment.' });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <RadioGroup
        value={state.experimentMode}
        onValueChange={(v) => patch({ experimentMode: v as WizardState['experimentMode'], resolved: null })}
        className="space-y-2"
      >
        <div className="flex items-center gap-2">
          <RadioGroupItem value="validate" id="mode-validate" />
          <Label htmlFor="mode-validate">Use an existing (fresh) experiment</Label>
        </div>
        <div className="flex items-center gap-2">
          <RadioGroupItem value="create" id="mode-create" />
          <Label htmlFor="mode-create">Create a new experiment</Label>
        </div>
      </RadioGroup>

      {state.experimentMode === 'validate' ? (
        <div className="space-y-2">
          <Label htmlFor="exp-id">MLflow experiment id</Label>
          <div className="flex flex-wrap items-center gap-2">
            <Input
              id="exp-id"
              className="w-72"
              value={state.experimentIdInput}
              placeholder="e.g. 660599403165942"
              onChange={(e) => patch({ experimentIdInput: e.target.value, resolved: null })}
            />
            <Button onClick={() => void validate()} disabled={busy || !state.experimentIdInput.trim()}>
              {busy ? 'Validating…' : 'Validate freshness'}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Fresh = the experiment exists, has no prior traces, and is not already registered to another agent.
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          <Label htmlFor="exp-name">New experiment name</Label>
          <div className="flex flex-wrap items-center gap-2">
            <Input
              id="exp-name"
              className="w-72"
              value={state.experimentNameInput}
              placeholder="/Users/you/my-new-agent"
              onChange={(e) => patch({ experimentNameInput: e.target.value, resolved: null })}
            />
            <Button onClick={() => void create()} disabled={busy || !state.experimentNameInput.trim()}>
              {busy ? 'Creating…' : 'Create experiment'}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Creating requires the app service principal to have experiment-create authority. If it does not, this
            returns an honest error and the prerequisite — nothing is fabricated.
          </p>
        </div>
      )}

      <Message message={message} />
      {state.resolved?.fresh && (
        <Badge variant="outline" className="text-emerald-700 dark:text-emerald-300">
          Ready: experiment {state.resolved.experiment_id}
        </Badge>
      )}
    </div>
  );
}

interface RequirementsProps extends StepProps {
  requirements: RequirementsResponse | null;
  reqError: string | null;
}

// Page 2 — the FIXED goal set (no free text). Multi-select; each option shows its
// resolved scorer (a deterministic L0 metric, or a MemAlign judge — never a fake
// judge for latency/cost). The catalog comes from the Python engine.
function GoalsStep({ state, patch, requirements, reqError }: RequirementsProps) {
  if (reqError) return <p className="text-sm text-destructive">{reqError}</p>;
  if (!requirements) return <Skeleton className="h-40 w-full" />;
  return (
    <div className="space-y-3">
      {requirements.catalog.map((goal) => {
        const checked = state.goals.includes(goal.key);
        return (
          <label
            key={goal.key}
            htmlFor={`goal-${goal.key}`}
            className="flex items-start gap-3 rounded-md border p-3 cursor-pointer hover:bg-muted/40"
          >
            <Checkbox
              id={`goal-${goal.key}`}
              checked={checked}
              onCheckedChange={(c) =>
                patch({
                  goals: c === true ? toggleGoal(state.goals, goal.key) : state.goals.filter((g) => g !== goal.key),
                })
              }
            />
            <div className="space-y-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium">{goal.label}</span>
                <Badge variant={goal.requires_quality ? 'default' : 'outline'}>
                  {goal.requires_quality ? 'judged (needs labels)' : 'deterministic'}
                </Badge>
                <Badge variant="outline">scorer: {goal.scorer}</Badge>
              </div>
              <p className="text-sm text-muted-foreground">{goal.description}</p>
            </div>
          </label>
        );
      })}
    </div>
  );
}

// Page 3 — the REAL readiness gates for the chosen goal(s), from the Python engine
// (docs/ONBOARDING_WIZARD.md §60). Explicit acceptance is required before Finish.
function DataGateStep({ state, patch, requirements, reqError }: RequirementsProps) {
  if (reqError) return <p className="text-sm text-destructive">{reqError}</p>;
  if (!requirements) return <Skeleton className="h-40 w-full" />;
  // Two-tier: the whole data-gate view is rendered VERBATIM from the Python engine.
  // No threshold number or gate-bundle string is authored here — the overall note,
  // the per-gate `needed` text, and each goal's `summary` all come from ail.readiness
  // via the requirements response, so the UI can never drift from the real floors.
  const view = dataGateView(requirements);
  return (
    <div className="space-y-4">
      {view.summary && <p className="text-sm">{view.summary}</p>}

      <div className="rounded-md border p-3 space-y-2">
        <p className="text-sm font-medium">Gates for your goal(s)</p>
        <ul className="space-y-1 text-sm">
          {view.gates.map((gate) => (
            <li key={gate.name} className="text-muted-foreground">
              <span className="text-foreground">{gate.label}</span>: {gate.needed}
            </li>
          ))}
        </ul>
      </div>

      {view.perGoal.length > 0 && (
        <div className="space-y-2">
          {view.perGoal.map((goal) => (
            <div key={goal.key} className="text-xs text-muted-foreground">
              <span className="text-foreground font-medium">{goal.label}</span> ({goal.scorer}): {goal.summary}
            </div>
          ))}
        </div>
      )}

      <label htmlFor="accept-gates" className="flex items-start gap-3 rounded-md border p-3 cursor-pointer">
        <Checkbox id="accept-gates" checked={state.accepted} onCheckedChange={(c) => patch({ accepted: c === true })} />
        <span className="text-sm">
          I understand optimization will not act on this agent until these data gates are met.
        </span>
      </label>
    </div>
  );
}

// Page 4 — name + register. Registering reuses ail.publish_versions server-side, so
// the agent appears in the existing AgentSwitcher on the next refresh.
function RegisterStep({ state, patch, result }: StepProps & { result: RegisterResponse | null }) {
  const message = result ? registerMessage(result) : null;
  return (
    <div className="space-y-4">
      <div className="rounded-md border p-3 space-y-1 text-sm">
        <p>
          <span className="font-medium">Experiment:</span> {state.resolved?.experiment_id}
          {state.resolved?.name ? ` (${state.resolved.name})` : ''}
        </p>
        <p>
          <span className="font-medium">Goals:</span> {state.goals.join(', ') || '—'}
        </p>
      </div>
      <div className="space-y-2">
        <Label htmlFor="agent-name">Agent name (unique)</Label>
        <Input
          id="agent-name"
          className="w-72"
          value={state.agentName}
          placeholder="e.g. my_claude_code_agent"
          onChange={(e) => patch({ agentName: e.target.value })}
          disabled={result?.outcome === 'registered'}
        />
      </div>
      <Message message={message} />
    </div>
  );
}
