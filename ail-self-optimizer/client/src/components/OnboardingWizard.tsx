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
  Tabs,
  TabsList,
  TabsTrigger,
  Textarea,
} from '@databricks/appkit-ui/react';
import {
  WIZARD_STEPS,
  canAdvance,
  clampStep,
  confirmRequirementsBody,
  confirmRequirementsMessage,
  createExperimentBody,
  creationMessage,
  dataGateView,
  freshnessMessage,
  initialWizardState,
  isLastStep,
  previewRequirementsBody,
  previewRequirementsMessage,
  registerBody,
  registerMessage,
  requirementsBody,
  requirementsPlanView,
  resolvedFromCreation,
  resolvedFromValidation,
  stepValidation,
  toggleGoal,
  validateExperimentBody,
  type CreationResponse,
  type RegisterResponse,
  type RequirementsConfirmResponse,
  type RequirementsPreviewResponse,
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
  previewRequirements: '/api/onboarding/requirements/preview',
  confirmRequirements: '/api/onboarding/requirements/confirm',
} as const;

// The two intake modes — pick from the fixed goal catalog (the slice-1 stepper) or
// describe requirements in free text (slice 2). Additive: the catalog path is
// unchanged; requirements mode is a parallel flow behind a tab.
type OnboardingMode = 'catalog' | 'requirements';

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
  const [mode, setMode] = useState<OnboardingMode>('catalog');
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
            <CardDescription>
              {mode === 'catalog'
                ? WIZARD_STEPS[state.stepIndex].description
                : 'Describe what to improve in your own words — the engine extracts the dimensions, routes each to a judge or a deterministic metric, and composes the goal.'}
            </CardDescription>
          </div>
          <Button variant="ghost" onClick={onClose} aria-label="Close wizard">
            Close
          </Button>
        </div>
        <Tabs value={mode} onValueChange={(v) => setMode(v as OnboardingMode)}>
          <TabsList>
            <TabsTrigger value="catalog">Goal catalog</TabsTrigger>
            <TabsTrigger value="requirements">Describe requirements</TabsTrigger>
          </TabsList>
        </Tabs>
        {mode === 'catalog' && <Stepper stepIndex={state.stepIndex} />}
      </CardHeader>
      <CardContent className="space-y-5">
        {mode === 'catalog' ? (
          <>
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
                <Button
                  onClick={() => patch({ stepIndex: clampStep(state.stepIndex + 1) })}
                  disabled={!canAdvance(state)}
                >
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
          </>
        ) : (
          <RequirementsMode state={state} patch={patch} onClose={onClose} />
        )}
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

// Free-form requirements intake (slice 2). The user resolves a fresh experiment,
// names the agent, and describes requirements in their own words; the engine
// previews the routed plan, and — after the human sets/acknowledges the objective
// target — authors the judges + persists the goal. Every routing/kind/target fact
// is rendered from the Python response; nothing is re-derived here (two-tier).
function RequirementsMode({ state, patch, onClose }: StepProps & { onClose: () => void }) {
  const [requirementsText, setRequirementsText] = useState('');
  const [preview, setPreview] = useState<RequirementsPreviewResponse | null>(null);
  const [previewMsg, setPreviewMsg] = useState<ToneMessage | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [targetInput, setTargetInput] = useState('');
  const [confirmResult, setConfirmResult] = useState<RequirementsConfirmResponse | null>(null);
  const [confirming, setConfirming] = useState(false);

  const resolved = state.resolved;
  const agentName = state.agentName.trim();
  const confirmed = confirmResult?.outcome === 'requirements_confirmed';
  const canPreview = Boolean(resolved?.fresh) && Boolean(agentName) && Boolean(requirementsText.trim()) && !previewing;

  async function runPreview() {
    setPreviewing(true);
    setPreviewMsg(null);
    setConfirmResult(null);
    try {
      const { status, body } = await postJson<RequirementsPreviewResponse>(
        API.previewRequirements,
        previewRequirementsBody(requirementsText, agentName)
      );
      if (status === 401) {
        setPreview(null);
        setPreviewMsg({ tone: 'error', text: 'Sign in to preview requirements.' });
        return;
      }
      if (body.outcome === 'requirements_preview') {
        setPreview(body);
        // Pre-fill the editable target from Python's SUGGESTION — never a TS constant.
        setTargetInput(body.suggested_target ? String(body.suggested_target.value) : '');
        setPreviewMsg(null);
      } else {
        setPreview(null);
        setPreviewMsg(previewRequirementsMessage(body));
      }
    } catch {
      setPreview(null);
      setPreviewMsg({ tone: 'error', text: 'Network error previewing requirements.' });
    } finally {
      setPreviewing(false);
    }
  }

  const parsedTarget = Number(targetInput);
  const targetValid = targetInput.trim() !== '' && Number.isFinite(parsedTarget);
  const canConfirm = Boolean(preview) && targetValid && !confirming && !confirmed;

  async function runConfirm() {
    if (!resolved) return;
    setConfirming(true);
    try {
      const { status, body } = await postJson<RequirementsConfirmResponse>(
        API.confirmRequirements,
        confirmRequirementsBody(requirementsText, agentName, resolved.experiment_id, parsedTarget)
      );
      if (status === 401) {
        setConfirmResult({ outcome: 'error', error: 'Not authenticated — sign in to confirm.' });
        return;
      }
      setConfirmResult(body);
    } catch {
      setConfirmResult({ outcome: 'error', error: 'Network error confirming requirements.' });
    } finally {
      setConfirming(false);
    }
  }

  const view = preview ? requirementsPlanView(preview) : null;

  return (
    <div className="space-y-5">
      <ExperimentStep state={state} patch={patch} />

      <div className="space-y-2">
        <Label htmlFor="req-agent-name">Agent name (the cohort these judges + goal attach to)</Label>
        <Input
          id="req-agent-name"
          className="w-72"
          value={state.agentName}
          placeholder="e.g. my_claude_code_agent"
          onChange={(e) => patch({ agentName: e.target.value })}
          disabled={confirmed}
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="req-text">Your requirements</Label>
        <Textarea
          id="req-text"
          rows={4}
          value={requirementsText}
          placeholder="e.g. correctness matters most; never hallucinate a tool call; keep latency and cost low"
          onChange={(e) => setRequirementsText(e.target.value)}
          disabled={confirmed}
        />
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={() => void runPreview()} disabled={!canPreview}>
            {previewing ? 'Previewing…' : 'Preview plan'}
          </Button>
          {!resolved?.fresh && (
            <span className="text-sm text-muted-foreground">Resolve a fresh experiment above first.</span>
          )}
          {resolved?.fresh && !agentName && (
            <span className="text-sm text-muted-foreground">Name the agent to preview.</span>
          )}
        </div>
        <Message message={previewMsg} />
      </div>

      {view && (
        <RequirementsPlanPanel
          view={view}
          targetInput={targetInput}
          onTargetChange={setTargetInput}
          onConfirm={() => void runConfirm()}
          canConfirm={canConfirm}
          confirming={confirming}
          confirmResult={confirmResult}
          onClose={onClose}
          disabled={confirmed}
        />
      )}
    </div>
  );
}

// The routed-plan review panel — rendered PURELY from the preview view model. No
// routing/threshold/target is authored here: the objective, each dimension's
// role/kind/direction, the judges-to-author vs deterministic-metrics split, and the
// SUGGESTED target all come from Python. The target input is pre-filled from the
// suggestion and editable; the human's value is sent back verbatim on confirm.
function RequirementsPlanPanel({
  view,
  targetInput,
  onTargetChange,
  onConfirm,
  canConfirm,
  confirming,
  confirmResult,
  onClose,
  disabled,
}: {
  view: ReturnType<typeof requirementsPlanView>;
  targetInput: string;
  onTargetChange: (value: string) => void;
  onConfirm: () => void;
  canConfirm: boolean;
  confirming: boolean;
  confirmResult: RequirementsConfirmResponse | null;
  onClose: () => void;
  disabled: boolean;
}) {
  const confirmed = confirmResult?.outcome === 'requirements_confirmed';
  const message = confirmResult ? confirmRequirementsMessage(confirmResult) : null;
  return (
    <div className="space-y-4 rounded-md border p-3">
      <div>
        <p className="text-sm font-medium">Routed plan</p>
        <p className="text-xs text-muted-foreground">
          Objective: <span className="text-foreground font-medium">{view.objectiveMetric}</span> ({view.direction})
          {view.requiresQuality ? ' · needs human labels to align its judge(s)' : ''}
        </p>
      </div>

      <ul className="space-y-2">
        {view.dimensions.map((d) => (
          <li key={d.name} className="rounded-md border p-2 space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium">{d.name}</span>
              <Badge variant={d.role === 'objective' ? 'default' : 'outline'}>{d.role}</Badge>
              <Badge variant={d.kind === 'memalign_judge' ? 'default' : 'outline'}>
                {d.kind === 'memalign_judge' ? `judge: ${d.judge_name}` : `metric: ${d.metric}`}
              </Badge>
              <Badge variant="outline">{d.direction}</Badge>
              <span className="text-xs text-muted-foreground">priority {d.user_priority}</span>
            </div>
            <p className="text-sm text-muted-foreground">{d.description}</p>
          </li>
        ))}
      </ul>

      <div className="grid gap-1 text-xs text-muted-foreground sm:grid-cols-2">
        <div>
          <span className="text-foreground font-medium">Judges to author:</span>{' '}
          {view.judgesToAuthor.join(', ') || 'none'}
        </div>
        <div>
          <span className="text-foreground font-medium">Deterministic metrics:</span>{' '}
          {view.deterministicMetrics.join(', ') || 'none'}
        </div>
      </div>

      {view.describe && <p className="break-words font-mono text-xs text-muted-foreground">{view.describe}</p>}

      <div className="space-y-2">
        <Label htmlFor="obj-target">
          Objective target
          {view.suggestedTarget ? ` (${view.suggestedTarget.kind}) — suggested, adjust before confirming` : ''}
        </Label>
        <Input
          id="obj-target"
          className="w-40"
          type="number"
          step="0.05"
          value={targetInput}
          onChange={(e) => onTargetChange(e.target.value)}
          disabled={disabled}
        />
        <p className="text-xs text-muted-foreground">
          A signed relative fraction; its sign must match the objective direction ({view.direction}). The engine fails
          closed on a mismatch — nothing is authored or persisted.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={onConfirm} disabled={!canConfirm}>
          {confirming ? 'Confirming…' : 'Confirm & author judges'}
        </Button>
        {confirmed && (
          <Button variant="outline" onClick={onClose}>
            Done
          </Button>
        )}
      </div>
      <Message message={message} />
    </div>
  );
}
