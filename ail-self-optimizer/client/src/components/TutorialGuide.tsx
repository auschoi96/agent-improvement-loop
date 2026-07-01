import { useEffect, useState } from 'react';
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Progress,
  Separator,
} from '@databricks/appkit-ui/react';
import { requirementsBody, type RequirementsResponse, type Thresholds } from '../lib/onboarding';
import {
  LOOP_STAGES,
  MEASUREMENT_LAYERS,
  TUTORIAL_STEPS,
  clampTutorialStep,
  isFirstTutorialStep,
  isLastTutorialStep,
  readinessGateLines,
  thresholdsFromRequirements,
  tutorialProgressPct,
  type LoopStage,
  type TutorialStep,
} from '../lib/tutorial';

// Read-only source of the readiness floors: the SAME authenticated requirements route
// the wizard already calls (POST with an empty goal selection returns the catalog +
// the `thresholds` object). The tutorial only READS it — it introduces no write-path
// and no new action; the app's only write-path remains approvals.
const REQUIREMENTS_ENDPOINT = '/api/onboarding/requirements';

type ThresholdStatus = 'loading' | 'loaded' | 'unavailable';

// The in-app "How it works" guided tutorial (docs/GETTING_STARTED.md, in-product). A
// thin renderer over lib/tutorial.ts: the step model, navigation, and the
// readiness-gate view are pure functions there; this component only draws them and
// fetches the live threshold numbers. Mirrors OnboardingWizard's structure (stepper
// Card + Progress) so it sits naturally beside the "Add an agent" wizard.
export function TutorialGuide({ onClose }: { onClose: () => void }) {
  const [stepIndex, setStepIndex] = useState(0);
  const [thresholds, setThresholds] = useState<Thresholds | null>(null);
  const [status, setStatus] = useState<ThresholdStatus>('loading');

  // Fetch the readiness floors once, fail-closed: any non-ok / error / network
  // failure leaves `thresholds` null so the readiness step shows a neutral
  // placeholder instead of a fabricated number. No actor is sent — the server
  // resolves the authenticated identity (a 401 simply falls to the neutral state).
  useEffect(() => {
    let live = true;
    fetch(REQUIREMENTS_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requirementsBody([])),
    })
      .then(async (res) => {
        const body = (await res.json().catch(() => null)) as RequirementsResponse | null;
        if (!live) return;
        const resolved = res.ok ? thresholdsFromRequirements(body) : null;
        setThresholds(resolved);
        setStatus(resolved ? 'loaded' : 'unavailable');
      })
      .catch(() => {
        if (!live) return;
        setThresholds(null);
        setStatus('unavailable');
      });
    return () => {
      live = false;
    };
  }, []);

  const step = TUTORIAL_STEPS[clampTutorialStep(stepIndex)];

  return (
    <Card className="shadow-sm border-primary/30">
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div>
            <CardTitle>How it works</CardTitle>
            <CardDescription>{step.tagline}</CardDescription>
          </div>
          <Button variant="ghost" onClick={onClose} aria-label="Close tutorial">
            Close
          </Button>
        </div>
        <Stepper stepIndex={stepIndex} />
      </CardHeader>
      <CardContent className="space-y-5">
        <StepBody step={step} />

        {step.key === 'loop' && <LoopDiagram />}
        {step.key === 'loop' && <MeasurementLayers />}
        {step.key === 'readiness' && <ReadinessGates thresholds={thresholds} status={status} />}

        <Separator />

        <div className="flex flex-wrap items-center gap-3">
          <Button
            variant="outline"
            onClick={() => setStepIndex((i) => clampTutorialStep(i - 1))}
            disabled={isFirstTutorialStep(stepIndex)}
          >
            Back
          </Button>
          {isLastTutorialStep(stepIndex) ? (
            <Button onClick={onClose}>Done</Button>
          ) : (
            <Button onClick={() => setStepIndex((i) => clampTutorialStep(i + 1))}>Next</Button>
          )}
          <span className="text-sm text-muted-foreground">
            Step {clampTutorialStep(stepIndex) + 1} of {TUTORIAL_STEPS.length}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

function Stepper({ stepIndex }: { stepIndex: number }) {
  return (
    <div className="space-y-2 pt-2">
      <Progress value={tutorialProgressPct(stepIndex)} />
      <ol className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {TUTORIAL_STEPS.map((s, i) => (
          <li
            key={s.key}
            className={
              i === clampTutorialStep(stepIndex)
                ? 'font-semibold text-foreground'
                : i < clampTutorialStep(stepIndex)
                  ? 'text-emerald-700 dark:text-emerald-300'
                  : 'text-muted-foreground'
            }
          >
            {i + 1}. {s.title}
            {i < clampTutorialStep(stepIndex) ? ' ✓' : ''}
          </li>
        ))}
      </ol>
    </div>
  );
}

function StepBody({ step }: { step: TutorialStep }) {
  return (
    <div className="space-y-3">
      {step.body.map((paragraph) => (
        <p key={paragraph} className="text-sm text-foreground">
          {paragraph}
        </p>
      ))}
      {step.points && step.points.length > 0 && (
        <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
          {step.points.map((point) => (
            <li key={point}>{point}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

// Lightweight inline visual of the loop — styled chips + arrows, no diagram runtime.
// The human-approval stage is tinted with the primary color so the control point
// reads at a glance; the trailing note closes the loop back to your agent.
const STAGE_TONE: Record<LoopStage['role'], string> = {
  agent: 'border-border bg-muted/40',
  measure: 'border-border bg-muted/40',
  control: 'border-border bg-muted/40',
  human: 'border-primary bg-primary/10',
  apply: 'border-border bg-muted/40',
};

function LoopDiagram() {
  return (
    <div className="rounded-md border p-3">
      <div className="flex flex-wrap items-stretch gap-2">
        {LOOP_STAGES.map((stage, i) => (
          <div key={stage.key} className="flex items-center gap-2">
            {i > 0 && <span className="text-muted-foreground select-none">→</span>}
            <div className={`rounded-md border px-3 py-2 ${STAGE_TONE[stage.role]}`}>
              <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                {stage.label}
                {stage.role === 'human' && (
                  <Badge variant="outline" className="text-primary">
                    you
                  </Badge>
                )}
              </div>
              <div className="text-xs text-muted-foreground">{stage.detail}</div>
            </div>
          </div>
        ))}
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        ↺ A gated, approved change flows back to your agent — and stays revertible via the lineage trail.
      </p>
    </div>
  );
}

function MeasurementLayers() {
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
      {MEASUREMENT_LAYERS.map((layer) => (
        <div key={layer.key} className="rounded-md border p-3 space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-foreground">{layer.name}</span>
            <Badge variant="outline">{layer.tagline}</Badge>
          </div>
          <p className="text-xs text-muted-foreground">{layer.detail}</p>
        </div>
      ))}
    </div>
  );
}

// The readiness gates and what unlocks at each. Every threshold NUMBER is rendered
// verbatim from the Python `thresholds` object via readinessGateLines — never
// authored here. When the floors are unavailable (loading, unauthenticated, or an
// engine error) each row shows the neutral placeholder and an honest note, never a
// fabricated number.
function ReadinessGates({ thresholds, status }: { thresholds: Thresholds | null; status: ThresholdStatus }) {
  const lines = readinessGateLines(thresholds);
  return (
    <div className="space-y-3">
      <div className="space-y-2">
        {lines.map((line) => (
          <div key={line.key} className="flex flex-wrap items-baseline gap-x-2 gap-y-1 rounded-md border p-3">
            <span className="text-sm font-medium text-foreground">{line.title}</span>
            <Badge variant={line.loaded ? 'default' : 'outline'}>{line.requirement}</Badge>
            <p className="w-full text-xs text-muted-foreground">{line.unlocks}</p>
          </div>
        ))}
      </div>
      {status === 'loading' && (
        <p className="text-xs text-muted-foreground">Loading the live readiness floors from the engine…</p>
      )}
      {status === 'unavailable' && (
        <p className="text-xs text-amber-700 dark:text-amber-300">
          Live readiness floors are served by the engine (a signed-in workspace session). Showing the gate structure;
          the exact floors appear once connected — no number is invented here.
        </p>
      )}
      {status === 'loaded' && (
        <p className="text-xs text-muted-foreground">
          These floors are the code-enforced defaults, fetched live from the engine — not values baked into the app.
        </p>
      )}
    </div>
  );
}
