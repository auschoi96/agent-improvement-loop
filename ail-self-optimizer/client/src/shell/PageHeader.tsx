import { Fragment, type ReactNode } from 'react';
import { Link, useLocation } from 'react-router';
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@databricks/appkit-ui/react';
import { breadcrumbsForPath, agentSearch, navItemForPath } from '../lib/navigation';
import { useAgent } from '../context/agent-context';

// The consistent page header: a breadcrumb (always) plus a title/description and a
// right-aligned actions slot and an optional freshness/meta line. Title and
// description default from the current route's IA entry (single source in
// navigation.ts) — bare-panel routes render an <h1>; flow/help routes (showTitle:
// false) render breadcrumb only, letting their own Card header be the hero so the page
// never shows a duplicate title. Callers can still override any field.
export function PageHeader({
  title: titleProp,
  description: descriptionProp,
  actions,
  meta,
}: {
  title?: string;
  description?: string;
  actions?: ReactNode;
  meta?: ReactNode;
}) {
  const { pathname } = useLocation();
  const { selected } = useAgent();
  const item = navItemForPath(pathname);
  const title = titleProp ?? (item?.showTitle ? item.title : undefined);
  const description = descriptionProp ?? (item?.showTitle ? item.description : undefined);
  const crumbs = breadcrumbsForPath(pathname);
  // Carry the selected agent through breadcrumb links so navigating "up" keeps the
  // current agent selected (matches the sidebar links).
  const search = agentSearch(selected?.agent_name);

  return (
    <div className="space-y-2">
      <Breadcrumb>
        <BreadcrumbList>
          {crumbs.map((crumb, i) => {
            const isLast = i === crumbs.length - 1;
            return (
              <Fragment key={crumb.label}>
                <BreadcrumbItem>
                  {isLast || !crumb.path ? (
                    <BreadcrumbPage>{crumb.label}</BreadcrumbPage>
                  ) : (
                    <BreadcrumbLink asChild>
                      <Link to={{ pathname: crumb.path, search }}>{crumb.label}</Link>
                    </BreadcrumbLink>
                  )}
                </BreadcrumbItem>
                {!isLast && <BreadcrumbSeparator />}
              </Fragment>
            );
          })}
        </BreadcrumbList>
      </Breadcrumb>

      {(title || description || actions) && (
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            {title && <h1 className="text-2xl font-bold tracking-tight text-foreground">{title}</h1>}
            {description && <p className="max-w-3xl text-sm text-muted-foreground">{description}</p>}
          </div>
          {actions && <div className="flex flex-shrink-0 items-center gap-2">{actions}</div>}
        </div>
      )}

      {meta && <div className="text-xs text-muted-foreground tabular-nums">{meta}</div>}
    </div>
  );
}
