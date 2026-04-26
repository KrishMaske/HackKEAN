/**
 * Content data layer for SceneShift.
 * Mirrors Seamless's content.ts pattern — maps JSON records to bundled assets.
 * All data is imported at build time. Zero runtime API calls.
 */

import showsData from "./shows.json";

// ─── User Personas ──────────────────────────────────────────────────────────
export interface UserPersona {
  id: number;
  key: string;
  name: string;
  emoji: string;
  color: string;
  vibe: string;
  focus: string;
  tags: string[];
}

export const users: UserPersona[] = [
  {
    id: 1,
    key: "gym",
    name: "Gym Bro",
    emoji: "💪",
    color: "hsl(24, 95%, 53%)",
    vibe: "Fitness-first, gains-obsessed, always tracking macros.",
    focus: "Exercise equipment, protein supplements, athletic gear.",
    tags: ["Fitness", "Performance"],
  },
  {
    id: 2,
    key: "foodie",
    name: "Foodie",
    emoji: "🍕",
    color: "hsl(142, 71%, 45%)",
    vibe: "Flavor-chasing, recipe-collecting, farmers-market regular.",
    focus: "Artisan food products, kitchen tools, gourmet ingredients.",
    tags: ["Culinary", "Organic"],
  },
  {
    id: 3,
    key: "tech",
    name: "Tech Nerd",
    emoji: "💻",
    color: "hsl(217, 91%, 60%)",
    vibe: "Early adopter, spec-obsessed, always has the latest gadget.",
    focus: "Computers, displays, peripherals, vintage tech.",
    tags: ["Technology", "Gadgets"],
  },
];

// ─── Show Content ───────────────────────────────────────────────────────────
export interface ShowItem {
  id: string;
  title: string;
  subtitle: string;
  year: number;
  videoKey: string;
  thumbnailKey: string;
  description: string;
  match: number;
  rating: string;
  seasons: number;
  targetObject: string;
}

/** Raw show records from shows.json */
interface ShowRecord {
  id: string;
  title: string;
  subtitle: string;
  year: number;
  videoKey: string;
  thumbnailKey: string;
  description: string;
  match?: number;
  rating?: string;
  seasons?: number;
  targetObject: string;
}

/**
 * Map video keys to backend-served URLs.
 * In Seamless these are static Vite imports — we use the FastAPI static mount instead,
 * which avoids bundling large video files into the frontend build.
 */
const VIDEO_BASE = "http://localhost:8000";

export function getVideoUrl(showId: string): string {
  const videoMap: Record<string, string> = {
    stranger_things_83: `${VIDEO_BASE}/input/STRANGER_THINGS_CLIP.mp4`,
    the_office_05: `${VIDEO_BASE}/input/OFFICE_CLIP.mp4`,
    succession_20: `${VIDEO_BASE}/input/SUCCESSION_CLIP.mp4`,
  };
  return videoMap[showId] ?? videoMap["stranger_things_83"];
}

export function getMaskVideoUrl(showId: string): string {
  return `${VIDEO_BASE}/masks/${showId}_mask.mp4`;
}

export function getPreviewVideoUrl(showId: string): string {
  return `${VIDEO_BASE}/masks/${showId}_preview.mp4`;
}

/** Build all shows from the static JSON catalog. */
export function buildShows(shows: ShowRecord[]): ShowItem[] {
  return shows.map((s) => ({
    id: s.id,
    title: s.title,
    subtitle: s.subtitle,
    year: s.year,
    videoKey: s.videoKey,
    thumbnailKey: s.thumbnailKey,
    description: s.description,
    match: s.match ?? 95,
    rating: s.rating ?? "TV-14",
    seasons: s.seasons ?? 1,
    targetObject: s.targetObject,
  }));
}

// ─── Exports ────────────────────────────────────────────────────────────────
const allShows = buildShows(showsData as ShowRecord[]);
const featuredShow = allShows[0]!;

export { allShows, featuredShow };
