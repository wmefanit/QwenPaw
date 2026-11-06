import { getApiUrl } from "../../api/config";
import { buildAuthHeaders } from "../../api/authHeaders";

export type QueueSteeringResult = "accepted" | "idle";

export async function submitQueueSteering(
  chatId: string,
  agentId: string,
  input: Array<Record<string, unknown>>,
  signal?: AbortSignal,
): Promise<QueueSteeringResult> {
  const headers = buildAuthHeaders();
  headers["X-Agent-Id"] = agentId;
  const response = await fetch(getApiUrl("/console/chat/steer"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...headers,
    },
    body: JSON.stringify({ chat_id: chatId, input }),
    signal,
  });
  if (response.status === 409) return "idle";
  if (!response.ok) {
    throw new Error(`Queue steering failed (HTTP ${response.status})`);
  }
  const result = (await response.json()) as {
    accepted_client_message_ids?: string[];
  };
  const expected = input.map((message) => {
    const metadata = message.metadata;
    return typeof metadata === "object" && metadata !== null
      ? (metadata as Record<string, unknown>).qwenpaw_client_message_id
      : undefined;
  });
  if (
    !Array.isArray(result.accepted_client_message_ids) ||
    expected.some((id) => !id || !result.accepted_client_message_ids?.includes(String(id)))
  ) {
    throw new Error("Queue steering receipt mismatch");
  }
  return "accepted";
}
