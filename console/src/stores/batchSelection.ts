/**
 * Batch send selection: which queued messages may travel in one turn.
 *
 * Each entry is serialized as its own content group, so media, mentions, and
 * quotes remain independent instead of being flattened into the first SDK
 * query/file list. The first incompatible routing/context item is a hard FIFO
 * boundary.
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

/** Every queued item has an independently serializable message group. */
export function isBatchable(item: QueueItem): boolean {
  return Boolean(item);
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
export function selectPendingSteeringBatchAfterAccepted(
  queue: QueueItem[],
  agentId: string,
  quantity: "one" | "all" = "all",
): QueueItem[] {
  let prefixLength = 0;
  while (
    prefixLength < queue.length &&
    queue[prefixLength].status === "sending" &&
    queue[prefixLength].steeringAccepted
  ) {
    prefixLength += 1;
  }
  const firstPending = queue[prefixLength];
  const acceptedHead = queue[0];
  if (
    !acceptedHead ||
    !firstPending ||
    firstPending.status !== "pending" ||
    !hasSameRequestSemantics(acceptedHead, firstPending, agentId)
  )
    return [];

  const batch = [firstPending];
  if (quantity === "one") return batch;
  for (let index = prefixLength + 1; index < queue.length; index += 1) {
    const candidate = queue[index];
    if (
      candidate.status !== "pending" ||
      !isBatchable(candidate) ||
      !hasSameRequestSemantics(acceptedHead, candidate, agentId) ||
      !hasSameRequestSemantics(firstPending, candidate, agentId)
    )
      break;
    batch.push(candidate);
  }
  return batch;
}

export function selectBatchItems(
  queue: QueueItem[],
  item: QueueItem,
  agentId: string,
  quantity: "one" | "all" = "all",
): QueueItem[] {
  if (item.status !== "pending" || !isBatchable(item)) return [item];

  const start = queue.findIndex((entry) => entry.id === item.id);
  if (start !== 0) return [item];

  const batch = [queue[start]];
  if (quantity === "one") return batch;
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
