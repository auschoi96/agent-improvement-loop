import { type ComponentType, type ReactNode } from 'react';
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from '@databricks/appkit-ui/react';
import { type LucideProps } from 'lucide-react';

// A consistent, honest empty state built on AppKit's Empty primitive: an icon, a
// plain-spoken title, a description of WHY it's empty, and an optional next-step
// action — never a decorative blank.
export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon?: ComponentType<LucideProps>;
  title: string;
  description?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <Empty className="rounded-lg border border-dashed">
      <EmptyHeader>
        {Icon && (
          <EmptyMedia variant="icon">
            <Icon className="h-6 w-6" />
          </EmptyMedia>
        )}
        <EmptyTitle>{title}</EmptyTitle>
        {description && <EmptyDescription>{description}</EmptyDescription>}
      </EmptyHeader>
      {action && <EmptyContent>{action}</EmptyContent>}
    </Empty>
  );
}
