"use client";
import { useMemo, useState } from "react";
import type { Experiment } from "@/lib/experiment";
import { Icon, percent, SectionTitle, Tag } from "./ui";
const metrics = [
  { key: "accuracy", label: "Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1-score" },
] as const;

export function Performance({ data }: { data: Experiment }) {
  const [mode, setMode] = useState<"scores" | "latency">("scores");
  return (
    <section className="card performance-card">
      <SectionTitle title="Performance at a glance">
        <div className="segmented" aria-label="Tampilan performa">
          <button
            onClick={() => setMode("scores")}
            className={mode === "scores" ? "selected" : ""}
            aria-pressed={mode === "scores"}
          >
            Scores
          </button>
          <button
            onClick={() => setMode("latency")}
            className={mode === "latency" ? "selected" : ""}
            aria-pressed={mode === "latency"}
          >
            Latency
          </button>
        </div>
      </SectionTitle>
      <p className="card-subtitle">
        {mode === "scores"
          ? "Precision, recall & F1 menggunakan macro average."
          : "Wall-clock per ulasan · termasuk vectorization / jaringan."}
      </p>
      <div className="chart-legend">
        <span>
          <i className="classic-swatch" /> Classic ML
        </span>
        <span>
          <i className="llm-swatch" /> Gemini API
        </span>
      </div>
      {mode === "scores" ? (
        <div className="score-chart">
          <div className="chart-y-labels">
            <span>100%</span>
            <span>75%</span>
            <span>50%</span>
            <span>25%</span>
            <span>0%</span>
          </div>
          <div className="chart-body">
            <div className="chart-grid-lines">
              <i />
              <i />
              <i />
              <i />
              <i />
            </div>
            {metrics.map((metric) => (
              <div className="chart-group" key={metric.key}>
                <div className="bar-pair">
                  <div
                    className="chart-bar classic-bar"
                    style={{
                      height: `${data.classic.metrics[metric.key] * 100}%`,
                    }}
                  >
                    <span>{percent(data.classic.metrics[metric.key])}</span>
                  </div>
                  {data.llm.metrics ? (
                    <div
                      className="chart-bar llm-bar"
                      style={{
                        height: `${data.llm.metrics[metric.key] * 100}%`,
                      }}
                    >
                      <span>{percent(data.llm.metrics[metric.key])}</span>
                    </div>
                  ) : (
                    <div
                      className="chart-bar pending-bar"
                      title="Hasil LLM belum lengkap"
                    >
                      <span>—</span>
                    </div>
                  )}
                </div>
                <span className="chart-x-label">{metric.label}</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="latency-content">
          {[
            {
              name: "Classic ML",
              mean: data.classic.mean_latency_ms,
              p95: data.classic.p95_latency_ms,
              icon: "code" as const,
            },
            {
              name: "Gemini API",
              mean: data.llm.mean_latency_ms,
              p95: data.llm.p95_latency_ms,
              icon: "spark" as const,
            },
          ].map((m) => (
            <div className="latency-row" key={m.name}>
              <span className="round-icon">
                <Icon name={m.icon} />
              </span>
              <div>
                <strong>{m.name}</strong>
                <small>
                  p95:{" "}
                  {m.p95 == null ? "belum tersedia" : `${m.p95.toFixed(2)} ms`}
                </small>
              </div>
              <b>
                {m.mean?.toFixed(2) ?? "—"}
                <small>ms / review</small>
              </b>
            </div>
          ))}
          <p>
            Latensi LLM adalah waktu request asli. Membaca cache tidak dihitung
            sebagai inferensi.
          </p>
        </div>
      )}
      <div className="card-footnote">
        <Icon name="info" size={14} />
        {data.llm.metrics
          ? `Perbandingan pada ${data.split.test_rows} ulasan yang sama; bukan benchmark produksi.`
          : "Skor Gemini muncul setelah seluruh prediksi API selesai."}
      </div>
    </section>
  );
}

export function DatasetCard({
  data,
  onExplore,
}: {
  data: Experiment;
  onExplore: () => void;
}) {
  const ratio = (data.dataset.labels.positif / data.dataset.rows) * 100;
  return (
    <section className="card dataset-card">
      <SectionTitle title="The dataset">
        <Tag tone="neutral">Original</Tag>
      </SectionTitle>
      <p className="card-subtitle">Ulasan pelanggan e-commerce Indonesia</p>
      <div className="donut-wrap">
        <div
          className="donut"
          style={{
            background: `conic-gradient(#315d49 0% ${ratio}%, #bdcd9e ${ratio}% 100%)`,
          }}
          role="img"
          aria-label={`${data.dataset.labels.positif} positif dan ${data.dataset.labels.negatif} negatif`}
        >
          <div>
            <strong>{data.dataset.rows}</strong>
            <span>total reviews</span>
          </div>
        </div>
        <span className="donut-orbit" />
      </div>
      <div className="distribution-key">
        <div>
          <span>
            <i className="classic-swatch" />
            Positif
          </span>
          <strong>
            {data.dataset.labels.positif}
            <small>{ratio.toFixed(0)}%</small>
          </strong>
        </div>
        <div>
          <span>
            <i className="llm-swatch" />
            Negatif
          </span>
          <strong>
            {data.dataset.labels.negatif}
            <small>{(100 - ratio).toFixed(0)}%</small>
          </strong>
        </div>
      </div>
      <button className="card-link" onClick={onExplore}>
        Explore dataset <Icon name="arrow" size={17} />
      </button>
    </section>
  );
}

export function Predictions({
  data,
  compact = false,
  onExpand,
}: {
  data: Experiment;
  compact?: boolean;
  onExpand?: () => void;
}) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const [page, setPage] = useState(0);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const selected = data.predictions.find((row) => row.review_id === selectedId);
  const filtered = useMemo(
    () =>
      data.predictions.filter((row) => {
        const search = `${row.review_text} ${row.product_name} ${row.review_id}`
          .toLowerCase()
          .includes(query.toLowerCase());
        const match =
          filter === "all" ||
          (filter === "errors" &&
            (row.classic !== row.actual ||
              (row.llm !== null && row.llm !== row.actual))) ||
          (filter === "disagree" &&
            row.llm !== null &&
            row.classic !== row.llm) ||
          row.actual === filter;
        return search && match;
      }),
    [data, query, filter],
  );
  const pageSize = compact ? 4 : 8;
  const currentPage = Math.min(
    page,
    Math.max(0, Math.ceil(filtered.length / pageSize) - 1),
  );
  const visible = filtered.slice(
    currentPage * pageSize,
    (currentPage + 1) * pageSize,
  );
  return (
    <section className="card predictions-card">
      <SectionTitle
        title={
          compact
            ? "A closer look at the predictions"
            : "Every review, accounted for"
        }
      >
        {compact ? (
          <button className="text-button" onClick={onExpand}>
            View all {data.predictions.length} <Icon name="arrow" size={16} />
          </button>
        ) : (
          <a
            className="button secondary small"
            href="/artifacts/predictions.csv"
            download
          >
            <Icon name="download" size={15} /> Export CSV
          </a>
        )}
      </SectionTitle>
      <p className="card-subtitle">
        Label asli dan hasil inference pada shared test set. Klik ulasan untuk
        detail.
      </p>
      {!compact && (
        <div className="table-controls">
          <label className="search-input">
            <Icon name="search" size={17} />
            <input
              aria-label="Cari ulasan, produk, atau ID"
              placeholder="Cari ulasan, produk, atau ID…"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setPage(0);
              }}
            />
          </label>
          <select
            aria-label="Filter prediksi"
            value={filter}
            onChange={(e) => {
              setFilter(e.target.value);
              setPage(0);
            }}
          >
            <option value="all">Semua prediksi</option>
            <option value="errors">Ada kesalahan</option>
            <option value="disagree">Model berbeda pendapat</option>
            <option value="positif">Aktual positif</option>
            <option value="negatif">Aktual negatif</option>
          </select>
          <span className="result-count">{filtered.length} reviews</span>
        </div>
      )}
      <div className="table-scroll">
        <table className="reviews-table">
          <thead>
            <tr>
              <th scope="col">CUSTOMER REVIEW</th>
              <th scope="col">ACTUAL</th>
              <th scope="col">
                <i className="classic-swatch" /> CLASSIC ML
              </th>
              <th scope="col">
                <i className="llm-swatch" /> GEMINI
              </th>
              <th scope="col">
                <span className="sr-only">Lihat detail</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {visible.map((row) => (
              <tr key={row.review_id}>
                <td>
                  <button
                    className="review-open"
                    onClick={() => setSelectedId(row.review_id)}
                  >
                    <span>{row.review_text}</span>
                    <small>
                      #{String(row.review_id).padStart(3, "0")} <b>·</b>{" "}
                      {row.product_name}
                    </small>
                  </button>
                </td>
                <td>
                  <Label value={row.actual} />
                </td>
                <td>
                  <Label
                    value={row.classic}
                    wrong={row.classic !== row.actual}
                  />
                </td>
                <td>
                  <Label
                    value={row.llm}
                    wrong={row.llm !== null && row.llm !== row.actual}
                  />
                </td>
                <td>
                  <button
                    className="row-button"
                    onClick={() => setSelectedId(row.review_id)}
                    aria-label={`Detail ulasan ${row.review_id}`}
                  >
                    <Icon name="chevron" size={16} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!visible.length && (
          <div className="empty-table">
            <Icon name="search" size={27} />
            <h3>Tidak ada ulasan yang cocok.</h3>
            <p>Coba kata pencarian atau filter lain.</p>
            <button
              className="text-button"
              onClick={() => {
                setQuery("");
                setFilter("all");
              }}
            >
              Reset filter
            </button>
          </div>
        )}
      </div>
      {!compact && (
        <div className="pagination">
          <span>
            {filtered.length ? currentPage * pageSize + 1 : 0}–
            {Math.min((currentPage + 1) * pageSize, filtered.length)} dari{" "}
            {filtered.length} ulasan
          </span>
          <div>
            <button
              disabled={currentPage === 0}
              onClick={() => setPage(currentPage - 1)}
            >
              Sebelumnya
            </button>
            <span>
              {currentPage + 1} /{" "}
              {Math.max(1, Math.ceil(filtered.length / pageSize))}
            </span>
            <button
              disabled={(currentPage + 1) * pageSize >= filtered.length}
              onClick={() => setPage(currentPage + 1)}
            >
              Berikutnya
            </button>
          </div>
        </div>
      )}
      {selected && (
        <div
          className="review-detail"
          role="region"
          aria-label={`Detail ulasan ${selected.review_id}`}
        >
          <div className="section-title">
            <Tag tone="neutral">REVIEW #{selected.review_id}</Tag>
            <button
              className="icon-button"
              onClick={() => setSelectedId(null)}
              aria-label="Tutup detail ulasan"
            >
              ×
            </button>
          </div>
          <h3>{selected.product_name}</h3>
          <blockquote>“{selected.review_text}”</blockquote>
          <div className="detail-predictions">
            <div>
              <small>Label asli</small>
              <Label value={selected.actual} />
            </div>
            <div>
              <small>Classic ML</small>
              <Label value={selected.classic} />
              <p>
                Probabilitas kelas: {percent(selected.classic_probability)}
                <br />
                <span>Belum dikalibrasi sebagai confidence.</span>
              </p>
            </div>
            <div>
              <small>Gemini</small>
              <Label value={selected.llm} />
              <p>
                {selected.llm_latency_ms == null
                  ? "Menunggu inference API"
                  : `${selected.llm_latency_ms.toFixed(0)} ms · waktu request asli`}
              </p>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
function Label({
  value,
  wrong = false,
}: {
  value: string | null;
  wrong?: boolean;
}) {
  return value ? (
    <span className={`sentiment-label ${value} ${wrong ? "wrong" : ""}`}>
      <i />
      {value}
      {wrong && <span title="Berbeda dari label aktual">↗</span>}
    </span>
  ) : (
    <span className="pending-label">— Pending</span>
  );
}
