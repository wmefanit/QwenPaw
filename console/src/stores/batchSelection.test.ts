/**
 * Batch send selection rules: only contiguous messages with matching request
 * semantics may travel in one turn.
 */
import { describe, expect, it } from "vitest";
import { isBatchable, selectBatchItems } from "./batchSelection";
import type { QueueItem } from "./messageQueueStore";

function makeItem(overrides: Partial<QueueItem> = {}): QueueItem {
  return {
    id: "item-1",
    text: "hello",
    status: "pending",
    retryCount: 0,
    createdAt: 0,
    agentId: "agent-a",
    backendSessionId: "session-a",
    userId: "user-a",
    channel: "console",
    ...overrides,
  };
}

describe("isBatchable", () => {
  it("accepts a plain-text item", () => {
    expect(isBatchable(makeItem())).toBe(true);
  });

  it.each([
    ["attachments", { attachments: [{ path: "/tmp/a.txt" } as never] }],
    ["images", { images: [{ path: "/tmp/a.png" } as never] }],
    ["mentions", { mentions: [{ id: "user-1" } as never] }],
    ["quotes", { quote: { text: "quoted" } as never }],
  ])("rejects an item with %s", (_label, overrides) => {
    expect(isBatchable(makeItem(overrides))).toBe(false);
  });
});

describe("selectBatchItems", () => {
  it("collects a contiguous run of pending plain-text items", () => {
    const queue = [
      makeItem({ id: "a", text: "alpha" }),
      makeItem({ id: "b", text: "beta" }),
      makeItem({ id: "c", text: "gamma" }),
    ];

    const batch = selectBatchItems(queue, queue[0], "agent-a");

    expect(batch.map((entry) => entry.id)).toEqual(["a", "b", "c"]);
  });

  it.each([
    ["sending", { status: "sending" as const }],
    ["failed", { status: "failed" as const }],
    ["attachment", { attachments: [{ path: "/tmp/b.txt" } as never] }],
    ["image", { images: [{ path: "/tmp/b.png" } as never] }],
    ["mention", { mentions: [{ id: "user-1" } as never] }],
    ["quote", { quote: { text: "quoted" } as never }],
    ["another agent", { agentId: "agent-b" }],
    ["another session", { backendSessionId: "session-b" }],
    ["another user", { userId: "user-b" }],
    ["another channel", { channel: "other-channel" }],
    [
      "another approval level",
      { requestContext: { approval_level: "STRICT" } },
    ],
    [
      "another backend control",
      { requestContext: { backend_controls: { mode: "strict" } } },
    ],
    [
      "another biz parameter",
      { bizParams: { user_prompt_params: { tenant: "b" } } },
    ],
  ])("stops before %s instead of skipping it", (_label, barrier) => {
    const queue = [
      makeItem({ id: "a" }),
      makeItem({ id: "barrier", ...barrier }),
      makeItem({ id: "c" }),
    ];

    const batch = selectBatchItems(queue, queue[0], "agent-a");

    expect(batch.map((entry) => entry.id)).toEqual(["a"]);
  });

  it("does not batch a non-pending or non-text starting item", () => {
    const item = makeItem({
      id: "start",
      attachments: [{ path: "/tmp/a.txt" } as never],
    });
    const queue = [item, makeItem({ id: "next" })];

    expect(selectBatchItems(queue, item, "agent-a")).toEqual([item]);
  });

  it("allows equivalent context objects with different key order", () => {
    const first = makeItem({
      id: "a",
      requestContext: { approval_level: "STRICT", nested: { a: 1, b: 2 } },
      bizParams: { tenant: "x", feature: { a: true, b: false } },
    });
    const second = makeItem({
      id: "b",
      requestContext: { nested: { b: 2, a: 1 }, approval_level: "STRICT" },
      bizParams: { feature: { b: false, a: true }, tenant: "x" },
    });

    expect(selectBatchItems([first, second], first, "agent-a")).toEqual([
      first,
      second,
    ]);
  });

  it("batches a compatible prefix and leaves the next incompatible item", () => {
    const queue = [
      makeItem({ id: "a" }),
      makeItem({ id: "b" }),
      makeItem({ id: "strict", requestContext: { approval_level: "STRICT" } }),
      makeItem({ id: "d" }),
    ];

    const batch = selectBatchItems(queue, queue[0], "agent-a");

    expect(batch.map((entry) => entry.id)).toEqual(["a", "b"]);
  });

  it("does not batch a non-head retry across older pending items", () => {
    const earlier = makeItem({ id: "earlier" });
    const retry = makeItem({ id: "retry" });
    const later = makeItem({ id: "later" });

    expect(selectBatchItems([earlier, retry, later], retry, "agent-a")).toEqual(
      [retry],
    );
  });

  it("returns only the requested item if it is no longer in the queue", () => {
    const item = makeItem({ id: "retry" });
    const queue = [makeItem({ id: "other" })];

    expect(selectBatchItems(queue, item, "agent-a")).toEqual([item]);
  });
});
