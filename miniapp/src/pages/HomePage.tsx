import { Settings, Globe, Bot, Wand2 } from "lucide-react";

interface Tile {
  id: string;
  name: string;
  desc: string;
  icon: React.ReactNode;
  enabled: boolean;
  onClick?: () => void;
}

interface HomePageProps {
  onNavigate: (page: "settings" | "skills") => void;
}

export function HomePage({ onNavigate }: HomePageProps) {
  const tiles: Tile[] = [
    {
      id: "settings",
      name: "Settings",
      desc: "Models, tools & rules",
      icon: <Settings size={24} color="#f59e0b" strokeWidth={2} />,
      enabled: true,
      onClick: () => onNavigate("settings"),
    },
    {
      id: "skills",
      name: "Skills",
      desc: "Manage & install skills",
      icon: <Wand2 size={24} color="#f59e0b" strokeWidth={2} />,
      enabled: true,
      onClick: () => onNavigate("skills"),
    },
    {
      id: "browser",
      name: "Browser",
      desc: "Coming soon",
      icon: <Globe size={24} color="#f59e0b" strokeWidth={2} />,
      enabled: false,
    },
    {
      id: "agent",
      name: "Chat",
      desc: "Coming soon",
      icon: <Bot size={24} color="#f59e0b" strokeWidth={2} />,
      enabled: false,
    },
  ];

  return (
    <div className="screen home-screen">
      <div className="home-greeting">
        <h1>AUG</h1>
        <p>Deus Ex Machina</p>
      </div>

      <div className="tile-grid">
        {tiles.map((tile) => (
          <div
            key={tile.id}
            className={`tile${tile.enabled ? "" : " tile--disabled"}`}
            onClick={tile.enabled ? tile.onClick : undefined}
          >
            <div className="tile-icon">{tile.icon}</div>
            <div className="tile-info">
              <div className="tile-name">{tile.name}</div>
              <div className="tile-desc">{tile.desc}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
