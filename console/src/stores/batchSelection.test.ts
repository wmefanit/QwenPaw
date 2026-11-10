/**
 * Batch send selection rules: only contiguous messages with matching request
 * semantics may travel in one turn.
 */
import { describe, expect, it } from "vitest";
import {
  isBatchable,
  selectBatchItems,
  selectPendingSteeringBatchAfterAccepted,
} from "./batchSelection";
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
  ])(
    "accepts an item with %s as an independent message group",
    (_label, overrides) => {
      expect(isBatchable(makeItem(overrides))).toBe(true);
    },
  );
});

describe("selectPendingSteeringBatchAfterAccepted", () => {
  it("selects only the compatible pending suffix after an accepted steer", () => {
    const queue = [
      makeItem({ id: "accepted-1", status: "sending", steeringAccepted: true }),
      makeItem({ id: "accepted-2", status: "sending", steeringAccepted: true }),
      makeItem({ id: "late-1" }),
      makeItem({ id: "late-2" }),
    ];
    expect(
      selectPendingSteeringBatchAfterAccepted(queue, "agent-a").map(
        (item) => item.id,
      ),
    ).toEqual(["late-1", "late-2"]);
  });

  it("preserves request-context and quantity boundaries for appended steering", () => {
    const queue = [
      makeItem({ id: "accepted", status: "sending", steeringAccepted: true }),
      makeItem({ id: "late-1" }),
      makeItem({
        id: "different-context",
        requestContext: { approval_level: "STRICT" },
      }),
      makeItem({ id: "late-3" }),
    ];
    expect(
      selectPendingSteeringBatchAfterAccepted(queue, "agent-a", "all").map(
        (item) => item.id,
      ),
    ).toEqual(["late-1"]);
    expect(
      selectPendingSteeringBatchAfterAccepted(queue, "agent-a", "one").map(
        (item) => item.id,
      ),
    ).toEqual(["late-1"]);
  });

  it("does not cross a failed item, routing boundary, or mismatched accepted prefix", () => {
    const accepted = makeItem({
      id: "accepted",
      status: "sending",
      steeringAccepted: true,
    });
    const pending = makeItem({ id: "late", agentId: "agent-b" });
    expect(
      selectPendingSteeringBatchAfterAccepted([accepted, pending], "agent-a"),
    ).toEqual([]);
    expect(
      selectPendingSteeringBatchAfterAccepted(
        [accepted, makeItem({ id: "failed", status: "failed" }), pending],
        "agent-a",
      ),
    ).toEqual([]);
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

  it("honors one-at-a-time quantity without skipping FIFO siblings", () => {
    const queue = [
      makeItem({ id: "a" }),
      makeItem({ id: "b" }),
      makeItem({ id: "c" }),
    ];

    expect(
      selectBatchItems(queue, queue[0], "agent-a", "one").map(
        (item) => item.id,
      ),
    ).toEqual(["a"]);
  });

  it.each([
    ["sending", { status: "sending" as const }],
    ["failed", { status: "failed" as const }],
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

  it("batches multimodal starting items as independent groups", () => {
    const item = makeItem({
      id: "start",
      attachments: [{ path: "/tmp/a.txt" } as never],
      mentions: [{ id: "user-1", name: "User 1" }],
      quote: { messageId: "m-1", text: "quoted" },
    });
    const next = makeItem({
      id: "next",
      images: [{ path: "/tmp/a.png" } as never],
    });

    expect(selectBatchItems([item, next], item, "agent-a")).toEqual([
      item,
      next,
    ]);
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
