// Presentation for the Python-decided VersionComparisonStatus (ail.publish_versions).
// The TRUST verdict is decided in Tier A Python and carried in the `status` column;
// the app only maps it to a label/tone — it never re-derives trust. Crucially, the
// "improvement cleared" green tone is reserved for `proven_improvement`: a version
// whose readiness wall has NOT cleared is shown caution/amber, never green.

export type StatusTone = 'positive' | 'caution' | 'neutral' | 'negative';

export interface StatusPresentation {
  label: string;
  tone: StatusTone;
  description: string;
}

export function presentStatus(status: string): StatusPresentation {
  switch (status) {
    case 'proven_improvement':
      return {
        label: 'Proven improvement',
        tone: 'positive',
        description: 'Objective improved, correctness held, and the readiness wall has cleared.',
      };
    case 'controlled_proof_collecting':
      return {
        label: 'Controlled proof · collecting',
        tone: 'caution',
        description:
          'Measured on the frozen suite (L1-gated) with correctness held. Organic readiness is still collecting — not yet shown as a cleared improvement.',
      };
    case 'regressed':
      return {
        label: 'Regressed',
        tone: 'negative',
        description: 'The candidate is not an improvement over the baseline.',
      };
    case 'collecting':
      return {
        label: 'Collecting',
        tone: 'neutral',
        description: 'Not enough data to claim an improvement yet.',
      };
    default:
      return { label: status, tone: 'neutral', description: '' };
  }
}

// Badge variant for a tone (AppKit Badge: default | secondary | outline | destructive).
export function toneBadgeVariant(tone: StatusTone): 'default' | 'secondary' | 'outline' | 'destructive' {
  switch (tone) {
    case 'positive':
      return 'default';
    case 'caution':
      return 'secondary';
    case 'negative':
      return 'destructive';
    default:
      return 'outline';
  }
}

// Tailwind classes for a banner of the given tone (border + subtle fill + text).
export function toneBannerClasses(tone: StatusTone): string {
  switch (tone) {
    case 'positive':
      return 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300';
    case 'caution':
      return 'border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300';
    case 'negative':
      return 'border-destructive/40 bg-destructive/10 text-destructive';
    default:
      return 'border-border bg-muted text-muted-foreground';
  }
}

// Direction color for a single measured metric delta. `improved` is the
// Python-computed strict move in the metric's own better direction; a tie/neutral
// is muted. This colors the FACTUAL direction of a measured delta — it is not the
// trust verdict (that is the status banner above).
export function deltaDirectionClass(improved: boolean, changed: boolean): string {
  if (!changed) return 'text-muted-foreground';
  return improved ? 'text-emerald-600 dark:text-emerald-400' : 'text-destructive';
}
