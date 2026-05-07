import { createWorkersAI } from "workers-ai-provider";
import { callable, routeAgentRequest } from "agents";
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
    
    // Get current stage to determine which tools to expose
    const stage = await this.ctx.storage.get("stage") ?? "gather";
    
    const result = streamText({
      model: workersai("@cf/meta/llama-3.1-70b-instruct"),
      system: `You are a friendly academic report assistant. Your job is to help students write structured academic reports.
      
      If the user says Hi, Hello, or anything similar, greet them and explain that you will help them write a structured academic report. The process has three stages: GATHER, OUTLINE, and WRITE.

      User is a student who needs help writing an academic report. You are an assistant that guides them through the report writing process.
      Greet the user and explain that you will help them write a structured academic report. The process has three stages: GATHER, OUTLINE, and WRITE.


      When user initiates a conversation, they want to write an academic report. The report writing process has three stages: GATHER, OUTLINE, and WRITE. Your job is to guide the user through these stages, asking questions to gather necessary information, generating an outline for confirmation, and then writing the report section by section based on the confirmed outline.
      When user first starts conversation, introduce yourself as an academic report assistant and explain the three stages of the report writing process. Then, start with the GATHER stage by asking the user for the topic of their report. After they provide the topic, ask for the type of report (e.g., research paper, lab report, etc.), the desired language (English or Türkçe), the expected length, and any special requirements they may have. Ask one question at a time and wait for their response before asking the next question.
      Once you have gathered enough information, transition to the OUTLINE stage. Generate a structured outline for the report, including section titles and brief descriptions of what each section will cover. Present this outline to the user and ask for their confirmation or any edits they would like to make. Save the confirmed outline for reference during the writing stage.
      The basic outline is as follows, but feel free to adapt it based on the report type and requirements:
      1. Title page: Include the report title, author's name, date, student number, link to project's github.
      2. Introduction: Introduction to the topic, background information, and the purpose of the report.
      You are an academic report assistant. You help students write structured academic reports.
      If it's a project Report: 
        3. Project Description: Describe the project, its goals, and the technologies used.
        4. System Architecture: Explain the architecture of the project, including diagrams if necessary.
        5. Implementation Details: Provide details on the implementation, challenges faced, and how they were overcome.
        6. Results and Evaluation: Present the results of the project and evaluate its success based on the initial goals.
      If it's a Research Report: 3. Literature Review: Summarize existing research related to the topic.
        3. Methodology: Explain the methods used to conduct the research or complete the project.
        4. Results: Present the findings of the research or project.
        5. Discussion: Discuss the implications of the results, limitations of the study, and potential future work.
        6. Conclusion: Summarize the main findings and their significance.

      You work in 3 stages:
      1. GATHER: Ask the user for topic, report type, language (English or Türkçe), length, and any special requirements. Ask one question at a time.
      2. OUTLINE: Once you have enough info, generate a structured outline with section titles and brief descriptions. Ask the user to confirm or edit it.
      3. WRITE: Write each section one by one. After each section, wait for user feedback before continuing.

      Start simple. Ask one question at a time. Never stop the conversation.`,
            
      // Prune old tool calls to save tokens on long conversations
      messages: pruneMessages({
        messages: inlineDataUrls(await convertToModelMessages(this.messages)),
        toolCalls: "before-last-2-messages"
      }),
      tools: {
        // MCP tools from connected servers
        ...mcpTools,

        // Only expose state tool after initial gathering
        ...(stage !== "gather" && {
          getReportState: tool({
            description: "Get the current stage, outline, and written sections",
            inputSchema: z.object({}),
            execute: async () => {
              const stage = await this.ctx.storage.get("stage") ?? "gather";
              const outline = await this.ctx.storage.get("outline") ?? null;
              const completedSections = await this.ctx.storage.get("completedSections") ?? [];
              return { stage, outline, completedSections };
            }
          })
        }),

        // Tool to set stage - available from outline stage onwards
        ...(stage !== "gather" && {
          setStage: tool({
            description: "Set the current stage of the report writing process.",
            inputSchema: z.object({
              stage: z.enum(["gather", "outline", "write"]).describe("Current stage"),
              completedSections: z.array(z.string()).describe("List of completed sections (for write stage)").optional()
            }),
            execute: async ({ stage, completedSections }) => {
              await this.ctx.storage.put("stage", stage);
              if (completedSections) {
                await this.ctx.storage.put("completedSections", completedSections);
              }
              return { stage };
            }
          })
        }),

        // Save outline - available after gathering
        ...(stage !== "gather" && {
          saveOutline: tool({
            description: "Save the confirmed report outline for later reference.",
            inputSchema: z.object({
              title: z.string(),
              sections: z.array(z.object({
                id: z.string(),
                title: z.string(),
                description: z.string()
              }))
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
              await this.ctx.storage.put(`section:${sectionId}`, { title, content });
              return { saved: true };
            }
          })
        }),
      },
      stopWhen: stepCountIs(5),
      abortSignal: options?.abortSignal
    });

    return result.toUIMessageStreamResponse();
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
