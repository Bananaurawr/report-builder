import { createWorkersAI } from "workers-ai-provider";
import { callable, routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  convertToModelMessages,
  pruneMessages,
  stepCountIs,
  streamText,
  tool,
  type ModelMessage
} from "ai";
import { z } from "zod";

/**
 * The AI SDK's downloadAssets step runs `new URL(data)` on every file
 * part's string data. Data URIs parse as valid URLs, so it tries to
 * HTTP-fetch them and fails. Decode to Uint8Array so the SDK treats
 * them as inline data instead.
 */
function inlineDataUrls(messages: ModelMessage[]): ModelMessage[] {
  return messages.map((msg) => {
    if (msg.role !== "user" || typeof msg.content === "string") return msg;
    return {
      ...msg,
      content: msg.content.map((part) => {
        if (part.type !== "file" || typeof part.data !== "string") return part;
        const match = part.data.match(/^data:([^;]+);base64,(.+)$/);
        if (!match) return part;
        const bytes = Uint8Array.from(atob(match[2]), (c) => c.charCodeAt(0));
        return { ...part, data: bytes, mediaType: match[1] };
      })
    };
  });
}

export class ChatAgent extends AIChatAgent<Env> {
  maxPersistedMessages = 100;

  onStart() {
    // Configure OAuth popup behavior for MCP servers that require authentication
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const mcpTools = this.mcp.getAITools();
    const workersai = createWorkersAI({ binding: this.env.AI });
    
    const result = streamText({
      model: workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
      system: `You are an academic report assistant. You help students write structured academic reports.
        You work in 3 stages:
        1. GATHER: Ask the user for topic, report type, language (English or Türkçe), length, and any special requirements. Ask one question at a time.
        2. OUTLINE: Once you have enough info, generate a structured outline with section titles and brief descriptions. Ask the user to confirm or edit it.
        3. WRITE: Write each section one by one. After each section, wait for user feedback before continuing.

        Always track which stage you are in. Current stage is stored in context.`,
        
        // Prune old tool calls to save tokens on long conversations
      messages: pruneMessages({
        messages: inlineDataUrls(await convertToModelMessages(this.messages)),
        toolCalls: "before-last-2-messages"
      }),
      tools: {
        // MCP tools from connected servers
        ...mcpTools,

        // Tool to check which stage the workflow is in
        setStage: tool({
          description: "Set the current stage of the report writing process.",
          inputSchema: z.object({
            stage: z.enum(["gather", "outline", "write"]).describe("Current stage"),
            completedSections: z.array(z.string()).describe("List of completed sections (for write stage)").optional()
          }),
          execute: async ({ stage, completedSections }) => {
            // Save to Durable Object Storage for access in future messages
            await this.ctx.storage.put("stage", stage);
            if (completedSections) {
              await this.ctx.storage.put("completedSections", completedSections);
            }
            return { stage };
          }
        }),

        // Save the confirmed outline to storage so it can be referenced during the write stage
        saveOutline: tool({
          description: "Save the confirmed report outline for later reference.",
          inputSchema: z.object({
            title:z.string(),
            sections: z.array(z.object({
              id: z.string(),
              title: z.string(),
              description: z.string()
            }))
          }),
          execute: async ({ title, sections }) => {
            await this.ctx.storage.put("outline", { title, sections });
            return {saved:true, sectionCount:sections.length};
          }
        }),

        // Save a completed section
        saveSection: tool({
          description: "Save a written section to storage",
          inputSchema: z.object({
            sectionId: z.string(),
            title: z.string(),
            content: z.string()
          }),
          execute: async ({ sectionId, title, content }) => {
            await this.ctx.storage.put(`section:${sectionId}`, { title, content });
            return { saved: true };
          }
        }),

        // Get current report stage
        getReportState: tool({
          description: "Get the current stage, outline, and written sections",
          inputSchema: z.object({}),
          execute: async () => {
            const stage = await this.ctx.storage.get("stage") ?? "gather";
            const outline = await this.ctx.storage.get("outline") ?? null;
            const completedSections = await this.ctx.storage.get("completedSections") ?? [];
            return { stage, outline, completedSections };
          }
        }),

        // Approval tool: requires user confirmation before executing
        calculate: tool({
          description:
            "Perform a math calculation with two numbers. Requires user approval for large numbers.",
          inputSchema: z.object({
            a: z.number().describe("First number"),
            b: z.number().describe("Second number"),
            operator: z
              .enum(["+", "-", "*", "/", "%"])
              .describe("Arithmetic operator")
          }),
          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
            const ops: Record<string, (x: number, y: number) => number> = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y
            };
            if (operator === "/" && b === 0) {
              return { error: "Division by zero" };
            }
            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b)
            };
          }
        })
      },
      stopWhen: stepCountIs(5),
      abortSignal: options?.abortSignal
    });

    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    // Do the actual work here (send email, call API, etc.)
    console.log(`Executing scheduled task: ${description}`);

    // Notify connected clients via a broadcast event.
    // We use broadcast() instead of saveMessages() to avoid injecting
    // into chat history — that would cause the AI to see the notification
    // as new context and potentially loop.
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
