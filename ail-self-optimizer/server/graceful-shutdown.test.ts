import { EventEmitter } from 'node:events';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { installGracefulShutdownBackstop } from './graceful-shutdown';

class FakeProcess extends EventEmitter {
  exit = vi.fn();
}

afterEach(() => vi.useRealTimers());

describe('installGracefulShutdownBackstop', () => {
  it('forces a clean exit before the platform deadline', () => {
    vi.useFakeTimers();
    const target = new FakeProcess();
    installGracefulShutdownBackstop(target, 8_000);

    target.emit('SIGTERM');
    vi.advanceTimersByTime(7_999);
    expect(target.exit).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(target.exit).toHaveBeenCalledWith(0);
  });
});
