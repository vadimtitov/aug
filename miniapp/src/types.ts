export interface SkillSummary {
  name: string;
  description: string;
  always_on: boolean;
  file_count: number;
}

export interface SkillDetail {
  name: string;
  description: string;
  body: string;
  always_on: boolean;
  files: string[];
}

// From /api/v1/packages?family=skill
export interface ClawHubSkillCard {
  name: string;        // slug
  displayName: string;
  summary: string;
  ownerHandle: string;
  latestVersion: string; // semver string directly
  capabilityTags: string[];
  executesCode: boolean;
  isOfficial: boolean;
  updatedAt?: number;
}

// From /api/v1/skills/{slug}
export interface ClawHubSkillDetail {
  skill: {
    slug: string;
    displayName: string;
    summary: string;
    stats: {
      downloads: number;
      installsAllTime: number;
      installsCurrent: number;
      stars: number;
      comments: number;
      versions: number;
    };
    updatedAt?: number;
  };
  latestVersion: { version: string; changelog?: string };
  owner: { handle: string; displayName?: string };
  moderation: {
    verdict: string;
    isSuspicious: boolean;
    isMalwareBlocked: boolean;
  } | null;
}

export interface ClawHubListResponse {
  items: ClawHubSkillCard[];
  nextCursor: string | null;
}

export interface ClawHubSearchResult {
  score: number;
  slug: string;
  displayName: string;
  summary: string;
  version: string | null;
  updatedAt?: number;
}

// Navigation stack
export type PageState =
  | { page: "home" }
  | { page: "settings" }
  | { page: "skills" }
  | { page: "skill-detail"; skillName: string; source: "local" }
  | { page: "skill-detail"; skillName: string; source: "clawhub"; slug: string }
  | { page: "file-viewer"; skillName: string; filePath: string; source: "local" }
  | { page: "file-viewer"; skillName: string; filePath: string; source: "clawhub"; slug: string };
