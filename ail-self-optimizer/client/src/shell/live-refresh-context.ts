import { createContext, useContext } from 'react';

export const LiveRefreshContext = createContext(0);

export function useLiveRefreshRevision() {
  return useContext(LiveRefreshContext);
}
