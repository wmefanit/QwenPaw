/**
 * Batch send: the queue packs several plain-text messages into one SDK
 * submission via biz_params.qwenpaw_batch_input. These tests pin the unpacking
 * contract — one user message per entry, per-entry receipt ids, and the
 * private key never reaching the backend.
 */
import { describe, expect, it } from "vitest";
import {
  applyChatPayloadTransforms,
  BATCH_INPUT_KEY,
  expandBatchInput,
} from "./chatPayload";
import { QWENPAW_CLIENT_MESSAGE_ID_KEY } from "../../utils/clientMessageId";
import type { ChatRequestPayloadTransformItem } from "../../plugins/registry/types";

const template = {
  role: "user",
  content: [{ type: "text", text: "first" }],
  metadata: { existing: "kept" },
};

type TransformEntries = NonNullable<
  Parameters<typeof applyChatPayloadTransforms>[3]
>;

function testTransform(
  id: string,
  transform: ChatRequestPayloadTransformItem["transform"],
): TransformEntries {
  return [
    {
      registrationId: `test-plugin:${id}`,
      pluginId: "test-plugin",
      registeredAt: 0,
      item: { id, transform },
    },
  ];
}

function receiptId(message: Record<string, unknown>): string | undefined {
  const metadata = message.metadata as Record<string, unknown> | undefined;
  const value = metadata?.[QWENPAW_CLIENT_MESSAGE_ID_KEY];
  return typeof value === "string" ? value : undefined;
}

describe("applyChatPayloadTransforms batch receipt IDs", () => {
  it("rejects a transform that drops a batch user message", () => {
    expect(() =>
      applyChatPayloadTransforms(
        {
          input: [
            { role: "user", content: [{ type: "text", text: "alpha" }] },
            { role: "user", content: [{ type: "text", text: "beta" }] },
          ],
          session_id: "session-a",
        },
        "agent-a",
        undefined,
        testTransform("drop-input", ({ payload }) => ({
          ...payload,
          input: (payload.input as unknown[])[0],
        })),
        undefined,
        ["id-a", "id-b"],
      ),
    ).toThrow(/removed queued batch messages/);
  });

  it("restores each receipt id after a transform replaces user input objects", () => {
    const result = applyChatPayloadTransforms(
      {
        input: [
          { role: "user", content: [{ type: "text", text: "alpha" }] },
          { role: "assistant", content: [] },
          { role: "user", content: [{ type: "text", text: "beta" }] },
        ],
        session_id: "session-a",
      },
      "agent-a",
      undefined,
      testTransform("replace-input", ({ payload }) => ({
        ...payload,
        input: [
          { role: "user", content: [{ type: "text", text: "alpha" }] },
          { role: "assistant", content: [] },
          { role: "user", content: [{ type: "text", text: "beta" }] },
        ],
      })),
      undefined,
      ["id-a", "id-b"],
    );

    expect(
      (result.input as Array<Record<string, unknown>>).map(receiptId),
    ).toEqual(["id-a", undefined, "id-b"]);
  });
});

describe("expandBatchInput", () => {
  it("leaves a non-batch submission untouched", () => {
    const input = [template];
    const bizParams = { source: "console_chat" };

    const result = expandBatchInput(input, bizParams);

    expect(result.input).toBe(input);
    expect(result.bizParams).toBe(bizParams);
  });

  it("does not expand a single-entry batch", () => {
    const input = [template];
    const bizParams = {
      [BATCH_INPUT_KEY]: [{ text: "only", clientMessageId: "id-1" }],
    };

    const result = expandBatchInput(input, bizParams);

    expect(result.input).toBe(input);
    expect(result.bizParams).toBe(bizParams);
  });

  it("expands every entry into an independent user message", () => {
    const input = [template];
    const bizParams = {
      source: "console_chat",
      [BATCH_INPUT_KEY]: [
        { text: "alpha", clientMessageId: "id-a" },
        { text: "beta", clientMessageId: "id-b" },
        { text: "gamma", clientMessageId: "id-c" },
      ],
    };

    const { input: expanded } = expandBatchInput(input, bizParams);

    expect(expanded).toHaveLength(3);
    expect(expanded.map((entry) => entry.role)).toEqual([
      "user",
      "user",
      "user",
    ]);
    expect(
      expanded.map(
        (entry) => (entry.content as Array<{ text: string }>)[0].text,
      ),
    ).toEqual(["alpha", "beta", "gamma"]);
    expect(expanded.map(receiptId)).toEqual(["id-a", "id-b", "id-c"]);
  });

  it("keeps per-entry receipt ids distinct", () => {
    const input = [template];
    const bizParams = {
      [BATCH_INPUT_KEY]: [
        { text: "alpha", clientMessageId: "id-a" },
        { text: "beta", clientMessageId: "id-b" },
      ],
    };

    const { input: expanded } = expandBatchInput(input, bizParams);

    const ids = expanded.map(receiptId);
    expect(new Set(ids).size).toBe(2);
  });

  it("preserves the template metadata on every expanded message", () => {
    const input = [template];
    const bizParams = {
      [BATCH_INPUT_KEY]: [
        { text: "alpha", clientMessageId: "id-a" },
        { text: "beta", clientMessageId: "id-b" },
      ],
    };

    const { input: expanded } = expandBatchInput(input, bizParams);

    for (const entry of expanded) {
      expect((entry.metadata as Record<string, unknown>).existing).toBe("kept");
    }
  });

  it("strips the private batch key from biz_params", () => {
    const input = [template];
    const bizParams = {
      source: "console_chat",
      agent_id: "agent-a",
      [BATCH_INPUT_KEY]: [
        { text: "alpha", clientMessageId: "id-a" },
        { text: "beta", clientMessageId: "id-b" },
      ],
    };

    const { bizParams: cleaned } = expandBatchInput(input, bizParams);

    expect(cleaned).not.toHaveProperty(BATCH_INPUT_KEY);
    expect(cleaned).toEqual({ source: "console_chat", agent_id: "agent-a" });
  });

  it("ignores a malformed batch value", () => {
    const input = [template];
    const bizParams = { [BATCH_INPUT_KEY]: "not-an-array" };

    const result = expandBatchInput(input, bizParams);

    expect(result.input).toBe(input);
    expect(result.bizParams).toBe(bizParams);
  });

  it("ignores a batch when the submission has no input template", () => {
    const bizParams = {
      [BATCH_INPUT_KEY]: [
        { text: "alpha", clientMessageId: "id-a" },
        { text: "beta", clientMessageId: "id-b" },
      ],
    };

    const result = expandBatchInput([], bizParams);

    expect(result.input).toEqual([]);
    expect(result.bizParams).toBe(bizParams);
  });

  it("tolerates missing biz_params", () => {
    const input = [template];

    const result = expandBatchInput(input, undefined);

    expect(result.input).toBe(input);
    expect(result.bizParams).toBeUndefined();
  });
});
