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
  Textarea,
} from '@databricks/appkit-ui/react';
import {
  WIZARD_STEPS,
  canAdvance,
  clampStep,
  confirmRequirementsBody,
  confirmRequirementsMessage,
  createExperimentBody,
  creationDetails,
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
  requirementsForGoals,
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
import { postOnboardingJson } from '../lib/onboarding-api';

const API = {
  requirements: '/api/onboarding/requirements',
  validate: '/api/onboarding/experiment/validate',
  create: '/api/onboarding/experiment/create',
  register: '/api/onboarding/register',
  previewRequirements: '/api/onboarding/requirements/preview',
  confirmRequirements: '/api/onboarding/requirements/confirm',
} as const;

const TONE_CLASS: Record<Tone, string> = {
  success: 'text-emerald-700 dark:text-emerald-300',
  warning: 'text-amber-700 dark:text-amber-300',
  error: 'text-destructive',
  info: 'text-muted-foreground',
};

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

  // One Python response includes the catalog and every goal's computed gate bundle.
  // Checkbox changes are projected from that cache below, so Data Gates never waits
  // on another serverless onboarding job or briefly shows the previous selection.
  useEffect(() => {
    let live = true;
    postOnboardingJson<RequirementsResponse>(API.requirements, requirementsBody([]))
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
  }, []);

  const selectedRequirements = requirementsForGoals(requirements, state.goals);

  const validation = stepValidation(state);
  const stepKey = WIZARD_STEPS[state.stepIndex].key;

  function finish() {
    setRegisterResult(null);
    if (!state.resolved) return;
    void postOnboardingJson<RegisterResponse>(
      API.register,
      // Thread the captured executor path + memory table, and the requirements-confirmed
      // goal_config when present (null on the catalog path → RLM neutral). The actor is
      // NEVER sent — the server resolves it from the authenticated request.
      registerBody(state.agentName, state.resolved.experiment_id, state.goals, {
        targetWorkspace: state.targetWorkspace,
        annotationsTable: state.annotationsTable,
        goalConfig: state.goalConfig,
        reviewerExperimentId: state.reviewerExperimentId,
        optimizationTargetPath: state.optimizationTargetPath,
        optimizationValidationCommand: state.optimizationValidationCommand,
      })
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
        <>
          {stepKey === 'experiment' && <ExperimentStep state={state} patch={patch} />}
          {stepKey === 'goals' && (
            <GoalsStep state={state} patch={patch} requirements={selectedRequirements} reqError={reqError} />
          )}
          {stepKey === 'data_gate' && (
            <DataGateStep state={state} patch={patch} requirements={selectedRequirements} reqError={reqError} />
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
  // The last creation response, kept so the success view can surface the ready-to-use
  // deep-link + tracing snippet (both relayed VERBATIM from Python — never built here).
  const [created, setCreated] = useState<CreationResponse | null>(null);

  async function createReviewer(baseName: string): Promise<string> {
    const { body } = await postOnboardingJson<CreationResponse>(API.create, {
      ...createExperimentBody(`${baseName}-ail-internal`),
      allow_existing: true,
    });
    if (body.outcome !== 'created' || !body.experiment_id) {
      throw new Error(body.error ?? 'Could not create the isolated reviewer experiment.');
    }
    return body.experiment_id;
  }

  async function validate() {
    setBusy(true);
    setMessage(null);
    try {
      const { status, body } = await postOnboardingJson<ValidationResponse>(
        API.validate,
        validateExperimentBody(state.experimentIdInput)
      );
      if (status === 401) {
        setMessage({ tone: 'error', text: 'Sign in to validate an experiment.' });
        return;
      }
      setMessage(freshnessMessage(body));
      const resolved = resolvedFromValidation(body);
      if (resolved) {
        const reviewerExperimentId = await createReviewer(body.name || `agent-${body.experiment_id}`);
        patch({ resolved, reviewerExperimentId });
      } else {
        patch({ resolved: null, reviewerExperimentId: '' });
      }
    } catch (error) {
      setMessage({
        tone: 'error',
        text: error instanceof Error ? error.message : 'Network error validating the experiment.',
      });
    } finally {
      setBusy(false);
    }
  }

  async function create() {
    setBusy(true);
    setMessage(null);
    setCreated(null);
    try {
      const { status, body } = await postOnboardingJson<CreationResponse>(
        API.create,
        createExperimentBody(state.experimentNameInput)
      );
      if (status === 401) {
        setMessage({ tone: 'error', text: 'Sign in to create an experiment.' });
        return;
      }
      setMessage(creationMessage(body));
      setCreated(body);
      const resolved = resolvedFromCreation(body);
      if (resolved) {
        const reviewerExperimentId = await createReviewer(body.name || state.experimentNameInput);
        patch({
          resolved,
          reviewerExperimentId,
          annotationsTable: body.annotations_table ?? state.annotationsTable,
        });
      } else {
        patch({ resolved: null, reviewerExperimentId: '' });
      }
    } catch (error) {
      setMessage({
        tone: 'error',
        text: error instanceof Error ? error.message : 'Network error creating the experiment.',
      });
    } finally {
      setBusy(false);
    }
  }

  // Show the ready-to-use details only for the CURRENT creation: gate on the resolved
  // experiment (cleared on any mode/name edit), so a stale link never lingers.
  const details =
    created && state.resolved?.fresh && created.experiment_id === state.resolved.experiment_id
      ? creationDetails(created)
      : null;

  return (
    <div className="space-y-4">
      <RadioGroup
        value={state.experimentMode}
        onValueChange={(v) => patch({ experimentMode: v as WizardState['experimentMode'], resolved: null })}
        className="space-y-2"
      >
        <div className="flex items-center gap-2">
          <RadioGroupItem value="validate" id="mode-validate" />
          <Label htmlFor="mode-validate">Use an existing experiment and include its trace history</Label>
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
              {busy ? 'Validating…' : 'Validate & isolate reviewers'}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Existing traces are accepted when the experiment is not already registered. A separate reviewer experiment
            is created automatically.
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
      {details && (details.url || details.hint) && (
        <div className="space-y-2 rounded-md border p-3 text-sm">
          {details.url && (
            <p>
              <a href={details.url} target="_blank" rel="noreferrer" className="underline text-primary">
                Open the experiment in MLflow
              </a>
            </p>
          )}
          {details.hint && (
            <div className="space-y-1">
              <p className="text-muted-foreground">Point your agent&apos;s tracing here:</p>
              <pre className="overflow-x-auto rounded bg-muted p-2 text-xs">
                <code>{details.hint}</code>
              </pre>
            </div>
          )}
        </div>
      )}

      <Separator />

      <div className="space-y-2">
        <Label htmlFor="agent-name">Agent name</Label>
        <Input
          id="agent-name"
          className="w-72"
          value={state.agentName}
          placeholder="e.g. my_claude_code_agent"
          onChange={(e) => patch({ agentName: e.target.value })}
        />
        <p className="text-xs text-muted-foreground">
          The cohort name used for traces, judges, goals, and comparisons.
        </p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="target-workspace">Local project the companion may edit</Label>
        <Input
          id="target-workspace"
          value={state.targetWorkspace}
          onChange={(e) => patch({ targetWorkspace: e.target.value })}
          placeholder="/path/to/your/agent/repo"
        />
        <p className="text-xs text-muted-foreground">
          Optional at registration; the executor remains fail-closed until this path is configured.
        </p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="requirements-text">Your requirements</Label>
        <Textarea
          id="requirements-text"
          rows={4}
          value={state.requirementsText}
          placeholder="e.g. correctness matters most; never hallucinate a tool call; keep latency and cost low"
          onChange={(e) =>
            patch({
              requirementsText: e.target.value,
              goalConfig: null,
              customJudgeNames: [],
              accepted: false,
            })
          }
        />
        <p className="text-xs text-muted-foreground">
          In Goals, the engine will recommend deterministic metrics and turn subjective requirements into custom
          MemAlign judges for you to review before anything is authored.
        </p>
      </div>
    </div>
  );
}

interface RequirementsProps extends StepProps {
  requirements: RequirementsResponse | null;
  reqError: string | null;
}

// Page 2 — fixed goals plus natural-language custom goals. The Python requirements
// engine routes exact quantities to L0 metrics and subjective dimensions to MemAlign
// judges. Nothing is authored until the user reviews and confirms the routed plan.
function GoalsStep({ state, patch, requirements, reqError }: RequirementsProps) {
  const [preview, setPreview] = useState<RequirementsPreviewResponse | null>(null);
  const [previewMsg, setPreviewMsg] = useState<ToneMessage | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [targetInput, setTargetInput] = useState('');
  const [confirmResult, setConfirmResult] = useState<RequirementsConfirmResponse | null>(null);
  const [confirming, setConfirming] = useState(false);

  if (reqError) return <p className="text-sm text-destructive">{reqError}</p>;
  if (!requirements) return <Skeleton className="h-40 w-full" />;

  async function runPreview() {
    setPreviewing(true);
    setPreviewMsg(null);
    setConfirmResult(null);
    try {
      const { status, body } = await postOnboardingJson<RequirementsPreviewResponse>(
        API.previewRequirements,
        previewRequirementsBody(state.requirementsText, state.agentName)
      );
      if (status === 401) {
        setPreview(null);
        setPreviewMsg({ tone: 'error', text: 'Sign in to recommend custom goals.' });
      } else if (body.outcome === 'requirements_preview') {
        setPreview(body);
        setTargetInput(body.suggested_target ? String(body.suggested_target.value) : '');
      } else {
        setPreview(null);
        setPreviewMsg(previewRequirementsMessage(body));
      }
    } catch {
      setPreview(null);
      setPreviewMsg({ tone: 'error', text: 'Network error recommending custom goals.' });
    } finally {
      setPreviewing(false);
    }
  }

  const parsedTarget = Number(targetInput);
  const targetValid = targetInput.trim() !== '' && Number.isFinite(parsedTarget);

  async function runConfirm() {
    if (!state.resolved || !preview || !targetValid) return;
    setConfirming(true);
    try {
      const { status, body } = await postOnboardingJson<RequirementsConfirmResponse>(
        API.confirmRequirements,
        confirmRequirementsBody(state.requirementsText, state.agentName, state.resolved.experiment_id, parsedTarget)
      );
      if (status === 401) {
        setConfirmResult({ outcome: 'error', error: 'Not authenticated — sign in to confirm.' });
      } else {
        setConfirmResult(body);
        if (body.outcome === 'requirements_confirmed' && body.goal_config) {
          patch({
            goalConfig: body.goal_config,
            customJudgeNames: body.authored_judges ?? [],
            accepted: false,
          });
        }
      }
    } catch {
      setConfirmResult({ outcome: 'error', error: 'Network error confirming custom goals.' });
    } finally {
      setConfirming(false);
    }
  }

  const view = preview ? requirementsPlanView(preview) : null;
  const canPreview = Boolean(state.requirementsText.trim()) && !previewing;
  const canConfirm = Boolean(view) && targetValid && !confirming && state.goalConfig === null;

  return (
    <div className="space-y-5">
      <div className="space-y-3">
        <div>
          <p className="font-medium">Goal catalog</p>
          <p className="text-sm text-muted-foreground">Choose any built-in goals that apply.</p>
        </div>
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
                    accepted: false,
                  })
                }
              />
              <div className="space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{goal.label}</span>
                  <Badge variant={goal.requires_quality ? 'default' : 'outline'}>
                    {goal.requires_quality ? 'MemAlign · needs your labels' : 'deterministic'}
                  </Badge>
                  <Badge variant="outline">scorer: {goal.scorer}</Badge>
                </div>
                <p className="text-sm text-muted-foreground">{goal.description}</p>
              </div>
            </label>
          );
        })}
      </div>

      <Separator />

      <div className="space-y-3 rounded-md border p-4">
        <div>
          <p className="font-medium">Define custom goals from your requirements</p>
          <p className="text-sm text-muted-foreground">
            Subjective goals become custom MemAlign judges. They start uncalibrated: you must label traces with your own
            feedback before the app can align and trust them. Exact goals such as latency, cost, or tokens remain
            deterministic metrics.
          </p>
        </div>
        <Textarea
          rows={4}
          value={state.requirementsText}
          placeholder="Describe the behavior you want, such as: never claim a tool succeeded unless its output proves it."
          onChange={(e) => {
            setPreview(null);
            setConfirmResult(null);
            setTargetInput('');
            patch({ requirementsText: e.target.value, goalConfig: null, customJudgeNames: [], accepted: false });
          }}
        />
        <Button onClick={() => void runPreview()} disabled={!canPreview}>
          {previewing ? 'Recommending…' : 'Recommend & define goals'}
        </Button>
        <Message message={previewMsg} />
        {state.customJudgeNames.length > 0 && (
          <p className="text-sm text-emerald-700 dark:text-emerald-300">
            Custom judges authored: {state.customJudgeNames.join(', ')}. Add human feedback on the Labeling page to
            align them.
          </p>
        )}
        {view && (
          <RequirementsPlanPanel
            view={view}
            targetInput={targetInput}
            onTargetChange={setTargetInput}
            onConfirm={() => void runConfirm()}
            canConfirm={canConfirm}
            confirming={confirming}
            confirmResult={confirmResult}
            disabled={state.goalConfig !== null}
          />
        )}
      </div>
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

      {state.customJudgeNames.length > 0 && (
        <div className="rounded-md border border-amber-500/40 bg-amber-500/5 p-3 space-y-2">
          <p className="text-sm font-medium">Custom MemAlign judges need your feedback</p>
          <p className="text-sm text-muted-foreground">
            {state.customJudgeNames.join(', ')} will remain untrusted until you provide at least{' '}
            {requirements.thresholds.quality_min_labels} human labels and the scored coverage reaches{' '}
            {Math.round(requirements.thresholds.scored_coverage_floor * 100)}%. Label representative traces after
            registration; MemAlign uses that feedback to learn your standard.
          </p>
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
// the agent appears in the existing AgentSwitcher on the next refresh. This step also
// CAPTURES the two fields that make the agent fully functional across the loop — the
// executor's target workspace and the memory job's annotations table — and threads
// them (plus any requirements-confirmed goal_config) onto the register payload. Both
// are optional and are never fabricated: an agent registered without them is honestly
// registered-but-not-fully-functional (the executor / memory job stay fail-closed).
function RegisterStep({ state, patch, result }: StepProps & { result: RegisterResponse | null }) {
  const message = result ? registerMessage(result) : null;
  const done = result?.outcome === 'registered';
  return (
    <div className="space-y-4">
      <div className="rounded-md border p-3 space-y-1 text-sm">
        <p>
          <span className="font-medium">Experiment:</span> {state.resolved?.experiment_id}
          {state.resolved?.name ? ` (${state.resolved.name})` : ''}
        </p>
        <p>
          <span className="font-medium">Reviewer experiment:</span> {state.reviewerExperimentId || '—'}
        </p>
        <p>
          <span className="font-medium">Agent:</span> {state.agentName || '—'}
        </p>
        <p>
          <span className="font-medium">Goals:</span> {state.goals.join(', ') || '—'}
        </p>
        <p>
          <span className="font-medium">Custom judges:</span> {state.customJudgeNames.join(', ') || '—'}
        </p>
        <p>
          <span className="font-medium">Local project:</span> {state.targetWorkspace || 'not configured'}
        </p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="annotations-table">Annotations table (OTEL table the memory job reads)</Label>
        <Input
          id="annotations-table"
          className="w-full max-w-xl"
          value={state.annotationsTable}
          placeholder="e.g. catalog.schema.otel_annotations"
          onChange={(e) => patch({ annotationsTable: e.target.value })}
          disabled={done}
        />
        <p className="text-xs text-muted-foreground">
          Fully-qualified table the memory-distiller reads this agent&apos;s annotations from. Required for the memory
          job — it skips an agent with no annotations table. Leave blank to add it later. Not guessed.
        </p>
      </div>

      <div className="space-y-2 rounded-md border p-3">
        <p className="text-sm font-medium">Approved GEPA rewrite target (optional)</p>
        <Label htmlFor="optimization-target-path">Project-relative prompt or Claude skill file</Label>
        <Input
          id="optimization-target-path"
          className="w-full max-w-xl"
          value={state.optimizationTargetPath}
          placeholder=".claude/skills/my-agent/SKILL.md"
          onChange={(e) => patch({ optimizationTargetPath: e.target.value })}
          disabled={done}
        />
        <Label htmlFor="optimization-validation-command">Validation command</Label>
        <Input
          id="optimization-validation-command"
          className="w-full max-w-xl"
          value={state.optimizationValidationCommand}
          placeholder="python -m pytest -q"
          onChange={(e) => patch({ optimizationValidationCommand: e.target.value })}
          disabled={done}
        />
        <p className="text-xs text-muted-foreground">
          Both fields are required to enable the last mile. The hosted app only records the reviewed target; your local
          companion verifies the original hash, snapshots it, applies the exact MLflow artifact, runs this command, and
          rolls back on failure. Absolute paths and parent traversal are refused.
        </p>
      </div>

      <Message message={message} />
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
  onClose?: () => void;
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
        {confirmed && onClose && (
          <Button variant="outline" onClick={onClose}>
            Done
          </Button>
        )}
      </div>
      <Message message={message} />
    </div>
  );
}
