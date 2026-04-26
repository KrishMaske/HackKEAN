/**
 * Pre-computed agent reasoning results per (userId, showId).
 * Mirrors Seamless's contentProducts.ts pattern.
 * Each entry contains the full agent pipeline output: final selection,
 * reasoning log from all 4 agents, and product details.
 */

export interface AgentLogEntry {
  agent: string;
  action: string;
  message: string;
}

export interface ProductDetail {
  name: string;
  price: number;
  description: string;
  link: string;
  imageUrl: string;
  category: string;
}

export interface AgentResult {
  userId: number;
  showId: string;
  finalSelection: string;
  reasoningLog: AgentLogEntry[];
  product: ProductDetail;
}

import resultsJson from "./agentResults.json";

const entries = resultsJson as AgentResult[];
const resultsByUserAndShow = new Map<string, AgentResult>(
  entries.map((e) => [`${e.userId}-${e.showId}`, e])
);

/**
 * Returns the pre-computed agent result for the given show and user persona.
 * Different personas get completely different product selections and reasoning.
 * Returns null if no result is pre-computed for this combination.
 */
export function getAgentResultForShow(
  showId: string,
  userId: number
): AgentResult | null {
  const key = `${userId}-${showId}`;
  return resultsByUserAndShow.get(key) ?? null;
}

/**
 * Get all available agent results (useful for debugging/admin views).
 */
export function getAllAgentResults(): AgentResult[] {
  return entries;
}
