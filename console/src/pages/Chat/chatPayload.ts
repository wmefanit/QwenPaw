import { chatExtensions } from "../../plugins/registry/chatExtensions";
import { ChatList } from "../../plugins/registry/slotKeys";
import { attachClientMessageId } from "../../utils/clientMessageId";

/** Batch payload key carried through biz_params (never sent to backend). */
export const BATCH_INPUT_KEY = "qwenpaw_batch_input";

export interface BatchEntry {
  text: string;
  clientMessageId: string;
  content?: Array<Record<string, unknown>>;
  metadata?: Record<string, unknown>;
}

/**
 * Expand a packed batch into independent user messages.
 *
 * The SDK accepts a single query string, so the queue packs the whole batch
 * into ``biz_params.qwenpaw_batch_input``. This unpacks it back into one user
 * message per entry and strips the private key so it never reaches the
 * backend. Without a batch the original input is returned untouched.
 */
export function expandBatchInput(
  input: Array<Record<string, unknown>>,
  bizParams: Record<string, unknown> | undefined,
): {
  input: Array<Record<string, unknown>>;
  bizParams: Record<string, unknown> | undefined;
} {
  const raw = bizParams?.[BATCH_INPUT_KEY];
  const entries = Array.isArray(raw) ? (raw as BatchEntry[]) : null;
  const template = input[0];
  if (!entries || entries.length === 0 || !template) {
    return { input, bizParams };
  }
  const expanded = entries.map((entry) => {
    const templateMetadata =
      typeof template.metadata === "object" && template.metadata !== null
        ? (template.metadata as Record<string, unknown>)
        : {};
    const message = {
      ...template,
      ...(entry.metadata
        ? { metadata: { ...templateMetadata, ...entry.metadata } }
        : {}),
      content: entry.content ?? [{ type: "text", text: entry.text }],
    };
    return attachClientMessageId(message, entry.clientMessageId);
  });
  const cleaned = bizParams
    ? Object.fromEntries(
        Object.entries(bizParams).filter(([key]) => key !== BATCH_INPUT_KEY),
      )
    : bizParams;
  return { input: expanded, bizParams: cleaned };
}

/** Both foreground SDK sends and host background sends use this pipeline. */
export function applyChatPayloadTransforms(
  payload: Record<string, unknown>,
  agentId: string,
  clientMessageId?: string,
  transforms = chatExtensions.getListSnapshot()[
    ChatList.requestPayloadTransforms
  ],
  controls?: Record<string, unknown>,
  clientMessageIds?: string[],
): Record<string, unknown> {
  let requestBody = payload;
  for (const entry of transforms
    .slice()
    .sort((a, b) => (a.item.order ?? 100) - (b.item.order ?? 100))) {
    const next = entry.item.transform({
      payload: requestBody,
      sessionId: String(requestBody.session_id || ""),
      selectedAgent: agentId,
    });
    if (next && typeof next === "object") requestBody = next;
  }
  // Keep receipt identities even when an extension replaces input records.
  if (clientMessageIds?.length && !Array.isArray(requestBody.input)) {
    throw new Error(
      "A chat payload transform removed queued batch messages; the batch was not sent.",
    );
  }
  if (Array.isArray(requestBody.input)) {
    const input = [...requestBody.input] as Array<Record<string, unknown>>;
    const ids = clientMessageIds?.length
      ? clientMessageIds
      : clientMessageId
      ? [clientMessageId]
      : [];
    const userCount = input.filter((entry) => entry?.role === "user").length;
    if (clientMessageIds?.length && userCount !== clientMessageIds.length) {
      throw new Error(
        "A chat payload transform changed the number of queued batch messages; the batch was not sent.",
      );
    }
    let userIndex = 0;
    for (let i = 0; i < input.length && userIndex < ids.length; i += 1) {
      if (input[i]?.role !== "user") continue;
      input[i] = attachClientMessageId(input[i], ids[userIndex]);
      userIndex += 1;
    }
    if (userIndex > 0) requestBody = { ...requestBody, input };
  }
  // Host approval/backend controls are frozen when the user submits. A
  // background handoff or a plugin replacement must not drop that policy.
  if (controls && (controls.approval_level || controls.backend_controls)) {
    const existing = requestBody.request_context;
    requestBody = {
      ...requestBody,
      request_context: {
        ...(existing && typeof existing === "object" ? existing : {}),
        ...(controls.approval_level
          ? { approval_level: controls.approval_level }
          : {}),
        ...(controls.backend_controls
          ? { backend_controls: controls.backend_controls }
          : {}),
      },
    };
  }
  return requestBody;
}
