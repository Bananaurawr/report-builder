import { createWorkersAI } from "workers-ai-provider";
import { callable, routeAgentRequest } from "agents";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  convertToModelMessages,
  pruneMessages,
  stepCountIs,
  streamText,
  tool,
  type ModelMessage,
  type UserModelMessage
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
        if (part.type !== "file") return part;

        const filePart = part as Record<string, unknown>;
        const data =
          typeof part.data === "string"
            ? part.data
            : typeof filePart.url === "string"
            ? filePart.url
            : undefined;

        if (typeof data !== "string") return part;

        try {
          const match = data.match(/^data:([^;]+);base64,(.+)$/);
          if (!match) return part;
          const bytes = Uint8Array.from(atob(match[2]), (c) => c.charCodeAt(0));
          return {
            ...part,
            data: bytes,
            mediaType: part.mediaType ?? match[1]
          };
        } catch (error) {
          console.error("Error processing data URL:", error);
          return part;
        }
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
    try {
      const mcpTools = this.mcp.getAITools();
      const workersai = createWorkersAI({ binding: this.env.AI });

      // Get current stage to determine which tools to expose
      const stage = (await this.ctx.storage.get("stage")) ?? "gather";

      const result = streamText({
        model: workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
        system: `You are a friendly academic report writing assistant.

        Your ONLY job is to help users write academic reports step by step.

        RULES:
          - NEVER say "input is not sufficient"
          - NEVER refuse a message
          - If the user says anything, respond helpfully and ask ONE question to move forward
          - Always be warm and friendly
          - When user sends an image, acknowledge it and ask how it relates to their report

        HOW TO RESPOND:
          - If user greets you → greet back and ask: "What topic would you like to write a report about?"
          - If user gives a topic → ask what type of report (research paper, project report, essay, etc.)
          - If user gives report type → ask preferred language (English or Türkçe)
          - If user gives language → ask expected length
          - If user sends an image → acknowledge it and ask: "How does this image relate to your report? Should I include it in a specific section?"
          - Once you have topic + type + language → generate a report outline and show it to the user
          - Once outline is confirmed → start writing the report section by section

        Start simple. Ask one question at a time. Never stop the conversation.`,

        // Prune old tool calls to save tokens on long conversations
        messages: pruneMessages({
          messages: inlineDataUrls(await convertToModelMessages(this.messages)).map(msg => {
            if (msg.role !== "user" || typeof msg.content === "string") return msg;
            return {
              ...msg,
              content: (msg.content as Array<{type: string}>).filter(part => part.type !== "file") as UserModelMessage["content"]
            };
          }),
          toolCalls: "before-last-2-messages"
        }),
        tools: {
          // MCP tools from connected servers
          ...mcpTools,

          // Only expose state tool after initial gathering
          ...(stage !== "gather" && {
            getReportState: tool({
              description:
                "Get the current stage, outline, and written sections",
              inputSchema: z.object({}),
              execute: async () => {
                const stage = (await this.ctx.storage.get("stage")) ?? "gather";
                const outline = (await this.ctx.storage.get("outline")) ?? null;
                const completedSections =
                  (await this.ctx.storage.get("completedSections")) ?? [];
                return { stage, outline, completedSections };
              }
            })
          }),

          // Tool to set stage - available from outline stage onwards
          ...(stage !== "gather" && {
            setStage: tool({
              description:
                "Set the current stage of the report writing process.",
              inputSchema: z.object({
                stage: z
                  .enum(["gather", "outline", "write"])
                  .describe("Current stage"),
                completedSections: z
                  .array(z.string())
                  .describe("List of completed sections (for write stage)")
                  .optional()
              }),
              execute: async ({ stage, completedSections }) => {
                await this.ctx.storage.put("stage", stage);
                if (completedSections) {
                  await this.ctx.storage.put(
                    "completedSections",
                    completedSections
                  );
                }
                return { stage };
              }
            })
          }),

          // Save outline - available after gathering
          ...(stage !== "gather" && {
            saveOutline: tool({
              description:
                "Save the confirmed report outline for later reference.",
              inputSchema: z.object({
                title: z.string(),
                sections: z.array(
                  z.object({
                    id: z.string(),
                    title: z.string(),
                    description: z.string()
                  })
                )
              }),
              execute: async ({ title, sections }) => {
                await this.ctx.storage.put("outline", { title, sections });
                return { saved: true, sectionCount: sections.length };
              }
            })
          }),

          // Save section - available only in write stage
          ...(stage === "write" && {
            saveSection: tool({
              description: "Save a written section to storage",
              inputSchema: z.object({
                sectionId: z.string(),
                title: z.string(),
                content: z.string()
              }),
              execute: async ({ sectionId, title, content }) => {
                await this.ctx.storage.put(`section:${sectionId}`, {
                  title,
                  content
                });
                return { saved: true };
              }
            })
          })
        },
        abortSignal: options?.abortSignal
      });

      return result.toUIMessageStreamResponse();
    } catch (error) {
      console.error("Error in onChatMessage:", error);
      // Return a simple error response
      return new Response(
        "Sorry, I encountered an error processing your message. Please try again.",
        {
          status: 500,
          headers: { "content-type": "text/plain" }
        }
      );
    }
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
