import type { Metrics } from "@/lib/experiment";
export type IconName =
  | "grid"
  | "database"
  | "rows"
  | "flask"
  | "file"
  | "arrow"
  | "download"
  | "refresh"
  | "external"
  | "check"
  | "search"
  | "chevron"
  | "code"
  | "spark"
  | "info"
  | "clock";
export function Icon({
  name,
  size = 19,
  className = "",
}: {
  name: IconName;
  size?: number;
  className?: string;
}) {
  const paths: Record<IconName, React.ReactNode> = {
    grid: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
      </>
    ),
    database: (
      <>
        <ellipse cx="12" cy="5" rx="8" ry="3" />
        <path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0" />
      </>
    ),
    rows: (
      <>
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <path d="M3 10h18M3 15h18M9 4v16" />
      </>
    ),
    flask: (
      <path d="M9 3h6M10 3v7L4 20c-.3.6 0 1 1 1h14c1 0 1.3-.4 1-1l-6-10V3M7 16h10" />
    ),
    file: (
      <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9zM14 3v6h6M8 13h8M8 17h6" />
    ),
    arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
    download: <path d="M12 3v12m-5-5 5 5 5-5M4 16v4h16v-4" />,
    refresh: (
      <path d="M20 7v5h-5M4 17v-5h5M6 7a7 7 0 0 1 12-2l2 3M4 16l2 3a7 7 0 0 0 12-2" />
    ),
    external: (
      <path d="M14 3h7v7m0-7L10 14M11 4H5a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h13a2 2 0 0 0 2-2v-6" />
    ),
    check: <path d="m5 12 4 4L19 6" />,
    search: (
      <>
        <circle cx="10" cy="10" r="6" />
        <path d="m15 15 6 6" />
      </>
    ),
    chevron: <path d="m9 5 7 7-7 7" />,
    code: <path d="m8 6-6 6 6 6m8-12 6 6-6 6m-3-15-2 18" />,
    spark: (
      <path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5zM20 2v4m-2-2h4" />
    ),
    info: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v6m0-10v1" />
      </>
    ),
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
    >
      {paths[name]}
    </svg>
  );
}
export const percent = (x: number | null | undefined, digits = 1) =>
  x == null ? "—" : `${(x * 100).toFixed(digits)}%`;
export function Tag({
  children,
  tone = "green",
}: {
  children: React.ReactNode;
  tone?: string;
}) {
  return <span className={`tag ${tone}`}>{children}</span>;
}
export function SectionTitle({
  eyebrow,
  title,
  children,
}: {
  eyebrow?: string;
  title: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="section-title">
      <div>
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h2>{title}</h2>
      </div>
      {children}
    </div>
  );
}
export function Stat({
  label,
  value,
  suffix,
  detail,
  icon,
  accent = false,
}: {
  label: string;
  value: string;
  suffix?: string;
  detail: string;
  icon: IconName;
  accent?: boolean;
}) {
  return (
    <div className={`stat-card ${accent ? "stat-accent" : ""}`}>
      <div className="stat-label">
        {label}
        <Icon name={icon} size={17} />
      </div>
      <div className="stat-value">
        {value}
        <span>{suffix}</span>
      </div>
      <div className="stat-detail">{detail}</div>
    </div>
  );
}
export function Matrix({
  metrics: value,
  name,
  type,
  completed = 0,
  expected = 0,
}: {
  metrics: Metrics | null;
  name: string;
  type: string;
  completed?: number;
  expected?: number;
}) {
  return (
    <section className="card matrix-card">
      <div className="matrix-heading">
        <span className={`small-model-icon ${type}`}>
          <Icon name={type === "classic" ? "code" : "spark"} size={17} />
        </span>
        <h3>{name}</h3>
      </div>
      {value ? (
        <>
          <div className="matrix-layout">
            <span className="matrix-axis-y">ACTUAL</span>
            <div className="matrix-table">
              <span />
              <span className="matrix-axis-label">Negatif</span>
              <span className="matrix-axis-label">Positif</span>
              {value.confusion_matrix.map((row, i) => (
                <div className="matrix-table-row" key={i}>
                  <span className="matrix-axis-label">
                    {i === 0 ? "Negatif" : "Positif"}
                  </span>
                  {row.map((number, j) => (
                    <div
                      key={j}
                      className={`matrix-cell ${i === j ? "diagonal" : "off-diagonal"}`}
                    >
                      <strong>{number}</strong>
                      <span>{i === j ? "correct" : "incorrect"}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
          <div className="matrix-axis-x">PREDICTED</div>
          <div className="matrix-result">
            <span>
              <i /> {value.correct} correct
            </span>
            <span>{value.total - value.correct} incorrect</span>
            <strong>{percent(value.accuracy)}</strong>
          </div>
        </>
      ) : (
        <div className="matrix-pending">
          <span className="pending-orb">
            <Icon name="spark" size={27} />
          </span>
          <h4>Waiting for the evidence</h4>
          <p>
            {completed}/{expected} prediksi tersimpan.
            <br />
            Lengkapi eksperimen API untuk
            <br />
            melihat confusion matrix.
          </p>
          <Tag tone="neutral">Belum dievaluasi</Tag>
        </div>
      )}
    </section>
  );
}
