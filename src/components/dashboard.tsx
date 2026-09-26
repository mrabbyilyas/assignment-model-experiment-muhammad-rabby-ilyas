"use client";
import { useState } from "react";
import type { Experiment } from "@/lib/experiment";
import { Icon, percent, Tag, SectionTitle, Stat, Matrix } from "./ui";
import { Performance, DatasetCard, Predictions } from "./explore";
import { DatasetView, Methodology, Report } from "./research";
import Features from "./features";
const tabs = [
  { id: "overview", label: "Overview", icon: "grid" },
  { id: "dataset", label: "Dataset", icon: "database" },
  { id: "predictions", label: "Prediction explorer", icon: "rows" },
  { id: "methodology", label: "Methodology", icon: "flask" },
  { id: "report", label: "Report & findings", icon: "file" },
] as const;
type Tab = (typeof tabs)[number]["id"];
export default function Dashboard({
  initialData,
}: {
  initialData: Experiment;
}) {
  const [data, setData] = useState(initialData);
  const [active, setActive] = useState<Tab>("overview");
  const [refreshing, setRefreshing] = useState(false);
  const [notice, setNotice] = useState("");
  async function refresh() {
    setRefreshing(true);
    setNotice("");
    try {
      const response = await fetch("/api/experiment", { cache: "no-store" });
      if (!response.ok)
        throw new Error(
          "Hasil belum tersedia. Jalankan eksperimen Python terlebih dahulu.",
        );
      setData(await response.json());
      setNotice("Hasil terbaru sudah dimuat.");
    } catch (e) {
      setNotice(e instanceof Error ? e.message : "Gagal memuat hasil.");
    } finally {
      setRefreshing(false);
    }
  }
  function navigate(tab: Tab) {
    setActive(tab);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  const complete = data.llm.status === "complete";
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Langsung ke konten
      </a>
      <aside className="sidebar">
        <button
          className="brand"
          onClick={() => navigate("overview")}
          aria-label="Sentiment Lab overview"
        >
          <span className="brand-mark">s.</span>
          <span>
            sentiment<span className="brand-light">lab</span>
            <small>AN EXPERIMENT IN UNDERSTANDING</small>
          </span>
        </button>
        <div className="workspace-label">
          WORKSPACE <span>01</span>
        </div>
        <nav aria-label="Navigasi eksperimen">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`nav-item ${active === tab.id ? "active" : ""}`}
              aria-current={active === tab.id ? "page" : undefined}
              onClick={() => navigate(tab.id)}
            >
              <Icon name={tab.icon} />
              {tab.label}
              {active === tab.id && <span className="nav-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-experiment">
          <div className="tiny-label">THE EXPERIMENT</div>
          <h3>
            Classic meets
            <br />
            contemporary.
          </h3>
          <p>
            Satu dataset. Dua pendekatan.
            <br />
            Keputusan berbasis bukti.
          </p>
          <div className="sidebar-models">
            <span>scikit-learn</span>
            <span>Gemini</span>
          </div>
          <div className="mini-progress">
            <i style={{ width: complete ? "100%" : "50%" }} />
          </div>
          <small>
            {complete
              ? "2 / 2 approaches evaluated"
              : "1 / 2 approaches evaluated"}
          </small>
        </div>
        <div className="sidebar-bottom">
          <span className="avatar">RI</span>
          <div>
            <strong>M. Rabby Ilyas</strong>
            <small>AI Engineering Bootcamp</small>
          </div>
          <span className="online-dot" />
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span> <strong>Model experiment</strong>
          </div>
          <div className="topbar-right">
            <span className="local-status">
              <i /> Local research workspace
            </span>
            <span className="topbar-separator" />
            <a
              href="https://github.com/mrabbyilyas/assignment-model-experiment-muhammad-rabby-ilyas"
              target="_blank"
              rel="noreferrer"
              className="repo-link"
            >
              Repository <Icon name="external" size={14} />
            </a>
          </div>
        </header>
        <main id="main-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                <span className="eyebrow-line" /> EXPERIMENT 001{" "}
                <span className="eyebrow-slash">/</span> SENTIMENT ANALYSIS
              </div>
              <h1>
                {active === "overview"
                  ? "Model comparison"
                  : tabs.find((t) => t.id === active)?.label}
              </h1>
              <p>
                {active === "overview"
                  ? "Dua pendekatan, satu pertanyaan: seberapa baik mereka memahami pelanggan?"
                  : active === "dataset"
                    ? "Kenali datanya, sebelum mempercayai angkanya."
                    : active === "predictions"
                      ? "Telusuri setiap ulasan, label asli, dan hasil prediksi kedua model."
                      : active === "methodology"
                        ? "Eksperimen yang adil dimulai dari metode yang bisa diperiksa."
                        : "Dari angka eksperimen menuju keputusan teknis."}
              </p>
            </div>
            <div className="heading-actions">
              <button
                className="icon-button"
                onClick={refresh}
                disabled={refreshing}
                aria-label="Refresh hasil eksperimen"
                title="Refresh hasil eksperimen"
              >
                <Icon name="refresh" className={refreshing ? "spinning" : ""} />
              </button>
              <a
                className="button primary"
                href="/artifacts/experiment_report.md"
                download
              >
                <Icon name="download" size={17} /> Export report
              </a>
            </div>
          </div>
          {notice && (
            <div className="notice" role="status">
              <Icon name="info" size={16} />
              {notice}
              <button
                onClick={() => setNotice("")}
                aria-label="Tutup pemberitahuan"
              >
                ×
              </button>
            </div>
          )}
          {active === "overview" && (
            <>
              <Hero data={data} />
              <div className="stat-grid">
                <Stat
                  label="CUSTOMER REVIEWS"
                  value={String(data.dataset.rows)}
                  detail={`${data.dataset.unique_texts} teks unik · Bahasa Indonesia`}
                  icon="database"
                />
                <Stat
                  label="HELD-OUT TEST SET"
                  value={String(data.split.test_rows)}
                  suffix="reviews"
                  detail={`${data.split.test_groups} grup teks · split tanpa overlap`}
                  icon="rows"
                />
                <Stat
                  label="CLASSIC MACRO-F1"
                  value={percent(data.classic.metrics.f1)}
                  detail={`${data.classic.metrics.correct} dari ${data.split.test_rows} prediksi benar`}
                  icon="flask"
                />
                <Stat
                  label="GEMINI MACRO-F1"
                  value={percent(data.llm.metrics?.f1)}
                  detail={
                    complete
                      ? `${data.llm.metrics?.correct} dari ${data.split.test_rows} prediksi benar`
                      : `${data.llm.completed}/${data.llm.expected} selesai · menunggu API`
                  }
                  icon="spark"
                  accent
                />
              </div>
              <div className="overview-grid">
                <Performance data={data} />
                <DatasetCard
                  data={data}
                  onExplore={() => navigate("dataset")}
                />
              </div>
              <div className="section-gap">
                <SectionTitle
                  eyebrow="BEYOND THE SCORE"
                  title="Where the models get it right"
                >
                  <Tag tone="neutral">Baris aktual → kolom prediksi</Tag>
                </SectionTitle>
                <div className="matrix-grid">
                  <Matrix
                    metrics={data.classic.metrics}
                    name="TF-IDF + Logistic Regression"
                    type="classic"
                  />
                  <Matrix
                    metrics={data.llm.metrics}
                    name="Google Gemini"
                    type="llm"
                    completed={data.llm.completed}
                    expected={data.llm.expected}
                  />
                  <div className="insight-card">
                    <span className="insight-icon">
                      <Icon name="info" size={23} />
                    </span>
                    <div className="eyebrow">A NOTE ON THE DATA</div>
                    <h3>
                      Perfect scores need
                      <br />
                      imperfect questions.
                    </h3>
                    <p>
                      Dataset ini punya {data.dataset.repeated_rows} baris
                      berulang. Pemisahan per teks unik mencegah model melihat
                      kalimat yang sama saat training dan evaluasi.
                    </p>
                    <button
                      className="text-button"
                      onClick={() => navigate("methodology")}
                    >
                      Lihat metodologi <Icon name="arrow" size={17} />
                    </button>
                  </div>
                </div>
              </div>
              <div className="section-gap">
                <Predictions
                  data={data}
                  compact
                  onExpand={() => navigate("predictions")}
                />
              </div>
              <div className="recommendation-strip">
                <span className="round-icon">
                  <Icon name="spark" />
                </span>
                <div>
                  <span className="eyebrow">FROM EXPERIMENT TO DECISION</span>
                  <h3>
                    {complete
                      ? "Angka punya cerita. Rekomendasi punya alasan."
                      : "Keputusan yang baik menunggu bukti yang lengkap."}
                  </h3>
                </div>
                <button
                  onClick={() => navigate("report")}
                  className="button secondary"
                >
                  Read findings <Icon name="arrow" size={17} />
                </button>
              </div>
            </>
          )}
          {active === "dataset" && <DatasetView data={data} />}
          {active === "predictions" && <Predictions data={data} />}
          {active === "methodology" && <Methodology data={data} />}
          {active === "report" && (
            <>
              <Report data={data} />
              <Features data={data} />
            </>
          )}
          <footer>
            <span>
              <span className="footer-dot" /> Made of real data, not
              assumptions.
            </span>
            <span>
              Muhammad Rabby Ilyas <span className="footer-divider">/</span>{" "}
              REWORK · AI Engineering
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}
function Hero({ data }: { data: Experiment }) {
  return (
    <section className="hero">
      <div className="hero-content">
        <div className="hero-eyebrow">
          <span /> A CONTROLLED COMPARISON
        </div>
        <h2>
          Same reviews.
          <br />
          Different <em>intelligence.</em>
        </h2>
        <p>
          Machine learning klasik bertemu large language model.
          <br />
          Diuji pada ulasan yang sama, dinilai dengan metrik yang sama.
        </p>
        <div className="hero-tags">
          <span>
            <Icon name="check" size={14} /> Shared test set
          </span>
          <span>
            <Icon name="check" size={14} /> No text leakage
          </span>
          <span>
            <Icon name="check" size={14} /> Seed 42
          </span>
        </div>
      </div>
      <div
        className="hero-visual"
        aria-label="Dua model diuji pada test set yang sama"
      >
        <div className="orbit orbit-one" />
        <div className="orbit orbit-two" />
        <div className="orbit orbit-three" />
        <div className="visual-label visual-label-top">
          TWO PATHS. ONE BENCHMARK.
        </div>
        <svg viewBox="0 0 460 245" className="branch-lines" aria-hidden="true">
          <path d="M50 123H125Q155 123 155 83V66H230M155 123v56h75M295 66h50q20 0 20 30v27h55M295 179h50q20 0 20-30v-26" />
          <circle cx="50" cy="123" r="4" />
          <circle cx="420" cy="123" r="4" />
        </svg>
        <div className="diagram-input">
          <Icon name="rows" size={24} />
          <span>
            {data.split.test_rows}
            <small>REVIEWS</small>
          </span>
        </div>
        <div className="model-node classic-node">
          <div className="model-node-icon">
            <Icon name="code" size={21} />
          </div>
          <div>
            <strong>Classic ML</strong>
            <small>TF-IDF + Logistic Regression</small>
          </div>
          <span className="node-indicator" />
        </div>
        <div className="model-node gemini-node">
          <div className="model-node-icon">
            <Icon name="spark" size={21} />
          </div>
          <div>
            <strong>Google Gemini</strong>
            <small>Zero-shot · API inference</small>
          </div>
          <span className="node-indicator" />
        </div>
        <div className="diagram-output">
          <Icon name="check" size={24} />
        </div>
        <div className="visual-label visual-label-bottom">
          TRAIN & PREDICT <span>vs.</span> PROMPT & CLASSIFY
        </div>
      </div>
    </section>
  );
}
