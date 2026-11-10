import { request } from "../../api/request";
import { useMessageQueueStore } from "../../stores/messageQueueStore";
import type { ChatHistory } from "../../api/types/chat";
import type { IAgentScopeRuntimeWebUIRunHandle } from "@agentscope-ai/chat";

/** A disconnected SDK Run intentionally remains non-terminal. Queue locks
 * belong to the mounted Chat scope, so they must not await that Run forever. */
export async function awaitQueueAcceptance(
  execution: Promise<IAgentScopeRuntimeWebUIRunHandle>,
  signal: AbortSignal,
) {
  const run = await awaitInChatScope(execution, signal);
  const session = await awaitInChatScope(run.session, signal);
  if (!session.resolved) {
    throw session.error || new Error("SDK chat session is not ready");
  }
  return awaitInChatScope(run.accepted, signal);
}

/** Release a view's send lock when it leaves; the backend run stays attached
 * to its Chat and is reconciled through history when that Chat is reopened. */
export function awaitInChatScope<T>(
  promise: Promise<T>,
  signal: AbortSignal,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      signal.removeEventListener("abort", onAbort);
      reject(new DOMException("Chat changed", "AbortError"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) onAbort();
    promise.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        if (signal.aborted) onAbort();
        else resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", onAbort);
        if (signal.aborted) onAbort();
        else reject(error);
      },
    );
  });
}

function waitForPoll(signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const finish = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", finish);
      resolve();
    };
    const timer = setTimeout(finish, 1000);
    signal.addEventListener("abort", finish, { once: true });
    if (signal.aborted) finish();
  });
}

/** Reconcile accepted queue receipts without waiting for the active run to end. */
export async function reconcileQueueReceipts(
  chatId: string,
  signal: AbortSignal,
  agentId: string,
  queueKey = chatId,
): Promise<"idle" | "running"> {
  const chat = await request<ChatHistory>(
    `/chats/${encodeURIComponent(chatId)}`,
    { headers: { "X-Agent-Id": agentId }, signal },
  );
  if (signal.aborted) return "running";
  useMessageQueueStore
    .getState()
    .reconcileHistory(queueKey, agentId, chat.messages || []);
  if (chat.status === "idle" || chat.status === "running") return chat.status;
  throw new Error("Unable to confirm chat status");
}

/** Only an explicit backend idle state permits a subsequent queued turn. */
export async function waitForChatIdle(
  chatId: string,
  signal: AbortSignal,
  agentId: string,
  queueKey = chatId,
): Promise<boolean> {
  while (!signal.aborted) {
    try {
      const chat = await request<ChatHistory>(
        `/chats/${encodeURIComponent(chatId)}`,
        { headers: { "X-Agent-Id": agentId }, signal },
      );
      if (signal.aborted) return false;
      useMessageQueueStore
        .getState()
        .reconcileHistory(queueKey, agentId, chat.messages || []);
      if (chat.status === "idle") return true;
      if (chat.status !== "running") {
        throw new Error("Unable to confirm chat status");
      }
    } catch (error) {
      if (signal.aborted) return false;
      throw error;
    }
    await waitForPoll(signal);
  }
  return false;
}

/** Call only while holding the queue's send lock. A persisted `sending`
 * marker is not a live sender: reconcile its receipt before allowing the
 * queue to continue. Unknown receipt must remain explicitly retryable rather
 * than being silently resent (or blocking the queue forever). */
export async function recoverSendingQueueHead(
  chatId: string,
  signal: AbortSignal,
  agentId: string,
  queueKey: string,
  errorMessage: string,
): Promise<void> {
  const initialQueue = useMessageQueueStore.getState().getQueue(queueKey);
  const firstNonSendingIndex = initialQueue.findIndex(
    (item) => item.status !== "sending",
  );
  const sendingBatch = initialQueue.slice(
    0,
    firstNonSendingIndex < 0 ? initialQueue.length : firstNonSendingIndex,
  );
  const head = sendingBatch[0];
  if (!head) return;
  let confirmedIdle = false;
  try {
    confirmedIdle = await waitForChatIdle(chatId, signal, agentId, queueKey);
  } catch (error) {
    if (signal.aborted) return;
    if (error instanceof Error) errorMessage = error.message;
  }
  const state = useMessageQueueStore.getState();
  if (signal.aborted || state.getRunState(queueKey) === "paused") return;

  // A status/history error is not evidence of non-delivery. Keep all steer
  // markers untouched until an authoritative idle response is available.
  if (!confirmedIdle) {
    const hasSteerIntent = sendingBatch.some((sentItem) => {
      const current = useMessageQueueStore
        .getState()
        .getQueue(queueKey)
        .find((item) => item.id === sentItem.id);
      return Boolean(current?.steeringAccepted || current?.steeringSubmitting);
    });
    if (hasSteerIntent) return;
  }

  // waitForChatIdle has now established an authoritative idle state and
  // reconciled history. Items with durable receipts are already removed; only
  // steer requests explicitly marked accepted/submitting may safely fall back
  // to normal queued delivery when their receipt is absent.
  const recoverableSteering = sendingBatch.filter((sentItem) => {
    const current = useMessageQueueStore
      .getState()
      .getQueue(queueKey)
      .find((item) => item.id === sentItem.id);
    return (
      current?.status === "sending" &&
      Boolean(current.steeringAccepted || current.steeringSubmitting)
    );
  });
  useMessageQueueStore.getState().recoverUnconsumedSteering(
    queueKey,
    recoverableSteering.map((item) => item.id),
  );

  // Remaining sending items have unknown ordinary-send receipts and must stay
  // explicitly retryable instead of being blindly resent or stuck forever.
  for (const sentItem of sendingBatch) {
    const current = useMessageQueueStore
      .getState()
      .getQueue(queueKey)
      .find((item) => item.id === sentItem.id);
    if (current?.status !== "sending") continue;
    useMessageQueueStore
      .getState()
      .setItemStatus(queueKey, sentItem.id, "failed", errorMessage);
  }
  if (sendingBatch.some((item) => item.id === state.currentSendingId)) {
    useMessageQueueStore.getState().setCurrentSendingId(null);
  }
}
