/**
 * Pre-computed scene segments per (userId, showId).
 * Mirrors Seamless's contentAdSegments.ts pattern.
 * Each segment defines a time range in the source video where a product is detected
 * and a mask/preview video is available for overlay.
 */

export interface SceneSegment {
  startTime: number;
  endTime: number;
  productName: string;
  maskVideoPath: string;
  previewVideoPath: string;
  color: string;
}

export interface SceneSegmentsEntry {
  userId: number;
  showId: string;
  segments: SceneSegment[];
}

import segmentsJson from "./sceneSegments.json";

const entries = segmentsJson as SceneSegmentsEntry[];
const segmentsByUserAndShow = new Map<string, SceneSegment[]>(
  entries.map((e) => [`${e.userId}-${e.showId}`, e.segments])
);

/**
 * Returns scene segments for the given show and user persona.
 * Different personas may see different segment colors/styling.
 * Returns an empty array if none are defined.
 */
export function getSegmentsForShow(
  showId: string,
  userId: number
): SceneSegment[] {
  const key = `${userId}-${showId}`;
  return segmentsByUserAndShow.get(key) ?? [];
}
