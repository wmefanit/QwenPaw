/**
 * Batch send selection: which queued messages may travel in one turn.
 *
 * The SDK exposes one query and one file list per submission, so only
 * contiguous plain-text messages with identical routing and request semantics
 * can be batched. The first incompatible queue item is a hard boundary.
 */
import type { QueueItem } from "./messageQueueStore";

/** Stable JSON serialization used for conservative request-context matching. */
function stableJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stableJson).join(",")}]`;
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableJson(record[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value) ?? String(value);
}

/** A queued item can join a batch only when it carries nothing but text. */
export function isBatchable(item: QueueItem): boolean {
  return (
    !item.attachments?.length &&
    !item.images?.length &&
    !item.mentions?.length &&
    !item.quote
  );
}

function hasSameRequestSemantics(
  left: QueueItem,
  right: QueueItem,
  fallbackAgentId: string,
): boolean {
  return (
    (left.agentId || fallbackAgentId) === (right.agentId || fallbackAgentId) &&
    (left.backendSessionId || "") === (right.backendSessionId || "") &&
    (left.userId || "default") === (right.userId || "default") &&
    (left.channel || "console") === (right.channel || "console") &&
    stableJson(left.requestContext) === stableJson(right.requestContext) &&
    stableJson(left.bizParams) === stableJson(right.bizParams)
  );
}

/**
 * Select the contiguous compatible batch beginning at ``item``.
 *
 * This preserves queue order: a failed, sending, non-text, differently routed,
 * or differently configured item stops the batch rather than being skipped.
 * The caller normally passes the queue head; an explicitly retried item may
 * be elsewhere, but batching will still not jump over following queue entries.
 */
export function selectBatchItems(
  queue: QueueItem[],
  item: QueueItem,
  agentId: string,
): QueueItem[] {
  if (item.status !== "pending" || !isBatchable(item)) return [item];

  const start = queue.findIndex((entry) => entry.id === item.id);
  if (start !== 0) return [item];

  const batch = [queue[start]];
  for (let index = start + 1; index < queue.length; index += 1) {
    const candidate = queue[index];
    if (
      candidate.status !== "pending" ||
      !isBatchable(candidate) ||
      !hasSameRequestSemantics(item, candidate, agentId)
    ) {
      break;
    }
    batch.push(candidate);
  }
  return batch;
}
