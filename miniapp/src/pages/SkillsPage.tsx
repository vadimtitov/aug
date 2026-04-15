import { useEffect, useRef, useState } from "react";
import { ChevronLeft, Search, X, Zap, Shield, Star, Download } from "lucide-react";
import {
  clawhubGetSkill,
  clawhubList,
  clawhubSearch,
  listSkills,
} from "../api.ts";
import type {
  ClawHubListResponse,
  ClawHubSearchResult,
  ClawHubSkillCard,
  PageState,
  SkillSummary,
} from "../types.ts";
import { onInstalledVersionChange } from "../lib/installedVersion.ts";

interface SkillStats {
  downloads: number;
  installsCurrent: number;
  stars: number;
}

// Session-scoped caches — persist across tab switches and back navigation
const _statsCache = new Map<string, SkillStats>();
let _trendingCache: { items: ClawHubSkillCard[]; nextCursor: string | null } | null = null;
let _lastTab: Tab = "mine";
let _lastQuery = "";
let _lastSearchResults: ClawHubSearchResult[] = [];

type Tab = "mine" | "clawhub";

interface Props {
  onBack: () => void;
  onNavigate: (state: PageState) => void;
}

export function SkillsPage({ onBack, onNavigate }: Props) {
  const [tab, setTab] = useState<Tab>(_lastTab);

  return (
    <div className="screen">
      <div className="page-header">
        <button className="back-btn" onClick={onBack}>
          <ChevronLeft size={20} />
          Back
        </button>
        <h1>Skills</h1>
      </div>

      <div className="tab-bar">
        <button
          className={`tab${tab === "mine" ? " tab--active" : ""}`}
          onClick={() => { _lastTab = "mine"; setTab("mine"); }}
        >
          Mine
        </button>
        <button
          className={`tab${tab === "clawhub" ? " tab--active" : ""}`}
          onClick={() => { _lastTab = "clawhub"; setTab("clawhub"); }}
        >
          ClawHub
        </button>
      </div>

      {tab === "mine" ? (
        <MineTab onNavigate={onNavigate} />
      ) : (
        <ClawHubTab onNavigate={onNavigate} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Mine tab
// ---------------------------------------------------------------------------

function MineTab({ onNavigate }: { onNavigate: (s: PageState) => void }) {
  const [skills, setSkills] = useState<SkillSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  function fetchSkills() {
    setLoading(true);
    listSkills()
      .then(setSkills)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    fetchSkills();
    // Re-fetch whenever a skill is installed from ClawHub
    return onInstalledVersionChange(fetchSkills);
  }, []);

  if (loading) return <LoadingState />;
  if (error) return <ErrorState message={error} />;

  if (skills.length === 0) {
    return (
      <div className="centered">
        <p style={{ color: "var(--hint)" }}>No skills yet.</p>
        <p style={{ color: "var(--hint)", fontSize: 13 }}>
          Browse ClawHub to install skills.
        </p>
      </div>
    );
  }

  return (
    <div className="skill-list">
      {skills.map((skill) => (
        <button
          key={skill.name}
          className="skill-card"
          onClick={() =>
            onNavigate({ page: "skill-detail", skillName: skill.name, source: "local" })
          }
        >
          <div className="skill-card-header">
            <span className="skill-card-name">{skill.name}</span>
            {skill.always_on && (
              <span className="skill-badge skill-badge--always-on">Always on</span>
            )}
          </div>
          <p className="skill-card-desc">{skill.description}</p>
          {skill.file_count > 0 && (
            <span className="skill-card-meta">
              {skill.file_count} file{skill.file_count !== 1 ? "s" : ""}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ClawHub tab
// ---------------------------------------------------------------------------

function ClawHubTab({ onNavigate }: { onNavigate: (s: PageState) => void }) {
  const [query, setQuery] = useState(_lastQuery);
  const [trending, setTrending] = useState<ClawHubSkillCard[]>(_trendingCache?.items ?? []);
  const [searchResults, setSearchResults] = useState<ClawHubSearchResult[]>(_lastSearchResults);
  const [nextCursor, setNextCursor] = useState<string | null>(_trendingCache?.nextCursor ?? null);
  const [loading, setLoading] = useState(_trendingCache === null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [installedSlugs, setInstalledSlugs] = useState<Set<string>>(new Set());
  const searchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedQuery = useRef(query); // track query at mount to skip re-fetching restored results

  useEffect(() => {
    listSkills()
      .then((skills) => setInstalledSlugs(new Set(skills.map((s) => s.name))))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (_trendingCache) return;
    clawhubList()
      .then((res: ClawHubListResponse) => {
        _trendingCache = { items: res.items, nextCursor: res.nextCursor };
        setTrending(res.items);
        setNextCursor(res.nextCursor);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (searchTimer.current) clearTimeout(searchTimer.current);
    if (!query.trim()) {
      if (query !== mountedQuery.current) setSearchResults([]);
      return;
    }
    // Skip re-fetching on mount if we restored previous results for this query
    if (query === mountedQuery.current && _lastSearchResults.length > 0) return;
    searchTimer.current = setTimeout(() => {
      setSearching(true);
      clawhubSearch(query.trim())
        .then((r) => { _lastSearchResults = r; setSearchResults(r); })
        .catch(() => { _lastSearchResults = []; setSearchResults([]); })
        .finally(() => setSearching(false));
    }, 250);
    return () => {
      if (searchTimer.current) clearTimeout(searchTimer.current);
    };
  }, [query]);

  function loadMore() {
    if (!nextCursor || loadingMore) return;
    setLoadingMore(true);
    clawhubList(nextCursor)
      .then((res: ClawHubListResponse) => {
        const newItems = [...trending, ...res.items];
        _trendingCache = { items: newItems, nextCursor: res.nextCursor };
        setTrending(newItems);
        setNextCursor(res.nextCursor);
      })
      .catch(() => {})
      .finally(() => setLoadingMore(false));
  }

  const isSearching = query.trim().length > 0;

  return (
    <div className="clawhub-tab">
      <div className="search-bar-wrapper">
        <Search size={16} className="search-icon" color="var(--hint)" />
        <input
          className="search-bar"
          placeholder="Search skills…"
          value={query}
          onChange={(e) => { _lastQuery = e.target.value; setQuery(e.target.value); }}
        />
        {query && (
          <button className="search-clear" onClick={() => { _lastQuery = ""; setQuery(""); }}>
            <X size={14} color="var(--hint)" />
          </button>
        )}
      </div>

      {error && <ErrorState message={error} />}

      {!error && !isSearching && (
        loading ? <LoadingState /> : (
          <>
            <p className="section-label">Latest</p>
            <div className="skill-list">
              {trending.map((skill) => (
                <ClawHubCard
                  key={skill.name}
                  skill={skill}
                  installed={installedSlugs.has(skill.name)}
                  onNavigate={onNavigate}
                />
              ))}
            </div>
            {nextCursor && (
              <div style={{ padding: "0 16px 24px" }}>
                <button
                  className="btn-secondary"
                  style={{ width: "100%" }}
                  onClick={loadMore}
                  disabled={loadingMore}
                >
                  {loadingMore ? "Loading…" : "Load more"}
                </button>
              </div>
            )}
          </>
        )
      )}

      {!error && isSearching && (
        searching ? <LoadingState /> : searchResults.length === 0 ? (
          <div className="centered">
            <p style={{ color: "var(--hint)" }}>No results for "{query}"</p>
          </div>
        ) : (
          <div className="skill-list">
            {searchResults.map((r) => (
              <SearchResultCard
                key={r.slug}
                result={r}
                installed={installedSlugs.has(r.slug)}
                onNavigate={onNavigate}
              />
            ))}
          </div>
        )
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Cards
// ---------------------------------------------------------------------------

function ClawHubCard({
  skill,
  installed,
  onNavigate,
}: {
  skill: ClawHubSkillCard;
  installed: boolean;
  onNavigate: (s: PageState) => void;
}) {
  const [stats, setStats] = useState<SkillStats | null>(_statsCache.get(skill.name) ?? null);
  const cardRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (_statsCache.has(skill.name) || !cardRef.current) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        observer.disconnect();
        clawhubGetSkill(skill.name)
          .then((d) => {
            const s: SkillStats = {
              downloads: d.skill.stats.downloads,
              installsCurrent: d.skill.stats.installsCurrent,
              stars: d.skill.stats.stars,
            };
            _statsCache.set(skill.name, s);
            setStats(s);
          })
          .catch(() => {});
      },
      { threshold: 0.1, rootMargin: "0px 0px 100px 0px" }
    );
    observer.observe(cardRef.current);
    return () => observer.disconnect();
  }, [skill.name]);

  return (
    <button
      ref={cardRef}
      className="skill-card"
      onClick={() =>
        onNavigate({
          page: "skill-detail",
          skillName: skill.name,
          source: "clawhub",
          slug: skill.name,
        })
      }
    >
      <div className="skill-card-header">
        <span className="skill-card-name">{skill.displayName}</span>
        {installed && (
          <span className="skill-badge skill-badge--installed">Installed</span>
        )}
        {skill.isOfficial && (
          <span className="skill-badge skill-badge--official">Official</span>
        )}
      </div>
      <p className="skill-card-desc">{skill.summary}</p>
      <div className="skill-card-stats">
        <span className="skill-stat">
          <span className="skill-stat-muted">by</span> {skill.ownerHandle}
        </span>
        {stats != null && stats.downloads > 0 && (
          <span className="skill-stat" title="Downloads">
            <Download size={11} /> {_fmt(stats.downloads)}
          </span>
        )}
        {stats != null && stats.stars > 0 && (
          <span className="skill-stat" title="Stars">
            <Star size={11} /> {stats.stars}
          </span>
        )}
        {skill.executesCode && (
          <span className="skill-stat skill-stat-warn" title="Executes code">
            <Zap size={11} /> code
          </span>
        )}
        {skill.capabilityTags.length > 0 && (
          <span className="skill-stat skill-stat-warn" title={skill.capabilityTags.join(", ")}>
            <Shield size={11} /> {skill.capabilityTags.length}
          </span>
        )}
      </div>
    </button>
  );
}

function SearchResultCard({
  result,
  installed,
  onNavigate,
}: {
  result: ClawHubSearchResult;
  installed: boolean;
  onNavigate: (s: PageState) => void;
}) {
  const [stats, setStats] = useState<SkillStats | null>(_statsCache.get(result.slug) ?? null);
  const ago = result.updatedAt ? _timeAgo(result.updatedAt) : null;
  const cardRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (_statsCache.has(result.slug) || !cardRef.current) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        observer.disconnect();
        clawhubGetSkill(result.slug)
          .then((d) => {
            const s: SkillStats = {
              downloads: d.skill.stats.downloads,
              installsCurrent: d.skill.stats.installsCurrent,
              stars: d.skill.stats.stars,
            };
            _statsCache.set(result.slug, s);
            setStats(s);
          })
          .catch(() => {});
      },
      { threshold: 0.1, rootMargin: "0px 0px 100px 0px" }
    );
    observer.observe(cardRef.current);
    return () => observer.disconnect();
  }, [result.slug]);

  return (
    <button
      ref={cardRef}
      className="skill-card"
      onClick={() =>
        onNavigate({
          page: "skill-detail",
          skillName: result.slug,
          source: "clawhub",
          slug: result.slug,
        })
      }
    >
      <div className="skill-card-header">
        <span className="skill-card-name">{result.displayName}</span>
        {installed && (
          <span className="skill-badge skill-badge--installed">Installed</span>
        )}
      </div>
      <p className="skill-card-desc">{result.summary}</p>
      <div className="skill-card-stats">
        {stats != null && stats.downloads > 0 && (
          <span className="skill-stat" title="Downloads">
            <Download size={11} /> {_fmt(stats.downloads)}
          </span>
        )}
        {stats != null && stats.stars > 0 && (
          <span className="skill-stat" title="Stars">
            <Star size={11} /> {stats.stars}
          </span>
        )}
        <span className="skill-stat skill-stat-mono">{result.slug}</span>
        {ago && <span className="skill-stat">{ago}</span>}
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

function LoadingState() {
  return (
    <div className="centered">
      <div className="spinner" />
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="centered">
      <p style={{ color: "var(--destructive)", fontSize: 14 }}>{message}</p>
    </div>
  );
}

function _fmt(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

function _timeAgo(ms: number): string {
  const diff = Date.now() - ms;
  const days = Math.floor(diff / 86_400_000);
  if (days < 1) return "today";
  if (days < 7) return `${days}d ago`;
  if (days < 30) return `${Math.floor(days / 7)}w ago`;
  if (days < 365) return `${Math.floor(days / 30)}mo ago`;
  return `${Math.floor(days / 365)}y ago`;
}
