"use client";
import type { Experiment } from "@/lib/experiment";
import { Icon, percent, Tag, SectionTitle, Stat } from "./ui";
import { DatasetCard } from "./explore";

export function DatasetView({ data }: { data: Experiment }) {
  return (
    <>
      <div className="stat-grid">
        <Stat
          label="ORIGINAL ROWS"
          value={String(data.dataset.rows)}
          detail="Semua baris dipertahankan"
          icon="database"
        />
        <Stat
          label="UNIQUE TEXTS"
          value={String(data.dataset.unique_texts)}
          detail="Unit pemisahan train / test"
          icon="rows"
        />
        <Stat
          label="REPEATED ROWS"
          value={String(data.dataset.repeated_rows)}
          detail="Tidak dihapus dari data sumber"
          icon="file"
        />
        <Stat
          label="TEXT OVERLAP"
          value="0"
          detail="Antara training dan test set"
          icon="check"
          accent
        />
      </div>
      <div className="dataset-view-grid">
        <DatasetCard
          data={data}
          onExplore={() =>
            window.open(data.dataset.source, "_blank", "noopener,noreferrer")
          }
        />
        <section className="card split-card">
          <SectionTitle
            eyebrow="REPRODUCIBLE BY DESIGN"
            title="Split the groups. Keep the rows."
          />
          <p>
            Pembagian 80:20 dilakukan pada teks unik, kemudian seluruh baris
            dalam setiap grup ikut ke split yang sama. Proporsi baris dapat
            berbeda.
          </p>
          <div className="split-visual">
            <div style={{ flex: data.split.train_rows }}>
              <strong>{data.split.train_rows}</strong>
              <span>training reviews</span>
            </div>
            <div style={{ flex: data.split.test_rows }}>
              <strong>{data.split.test_rows}</strong>
              <span>test reviews</span>
            </div>
          </div>
          <div className="split-details">
            <div>
              <span>Train groups</span>
              <strong>{data.split.train_groups} unique texts</strong>
            </div>
            <div>
              <span>Test groups</span>
              <strong>{data.split.test_groups} unique texts</strong>
            </div>
            <div>
              <span>Stratification</span>
              <strong>Sentiment label</strong>
            </div>
            <div>
              <span>Random state</span>
              <strong>42 · fixed</strong>
            </div>
            <div>
              <span>Test composition</span>
              <strong>
                {data.split.test_labels.negatif} negatif /{" "}
                {data.split.test_labels.positif} positif
              </strong>
            </div>
          </div>
          <a
            className="text-button"
            href="/artifacts/split_manifest.csv"
            download
          >
            Download split manifest <Icon name="download" size={16} />
          </a>
        </section>
      </div>
      <section className="card section-gap data-integrity">
        <SectionTitle title="The leakage we chose to prevent">
          <Tag tone="amber">Diagnostic only</Tag>
        </SectionTitle>
        <p>
          Replikasi split acak per baris dari starter menghasilkan accuracy{" "}
          <strong>{percent(data.diagnostic.accuracy)}</strong>. Tetapi{" "}
          <strong>
            {data.diagnostic.overlapping_test_rows} dari{" "}
            {data.diagnostic.test_rows}
          </strong>{" "}
          teks test sudah muncul saat training. Karena itu angka ini tidak
          dipakai sebagai hasil utama.
        </p>
        <div className="integrity-comparison">
          <div>
            <span>Random row split</span>
            <div className="integrity-track">
              <i
                style={{
                  width: `${(data.diagnostic.overlapping_test_rows / data.diagnostic.test_rows) * 100}%`,
                }}
              />
            </div>
            <strong>
              {data.diagnostic.overlapping_test_rows} overlapping reviews
            </strong>
          </div>
          <div>
            <span>Our grouped split</span>
            <div className="integrity-track safe" />
            <strong>
              <Icon name="check" size={15} /> 0 overlapping reviews
            </strong>
          </div>
        </div>
      </section>
      <section className="card section-gap source-card">
        <SectionTitle title="Original source & provenance" />
        <p>
          Dataset resmi assignment. Kolom: review_id, product_name, review_text,
          sentiment. CSV sumber tidak diubah.
        </p>
        <label>DATASET SHA-256</label>
        <code>{data.dataset.sha256}</code>
        <div className="download-row">
          <a
            className="button primary"
            href="/artifacts/customer_reviews_sentiment.csv"
            download
          >
            <Icon name="download" size={16} /> Original dataset
          </a>
          <a
            className="button secondary"
            href={data.dataset.source}
            target="_blank"
            rel="noreferrer"
          >
            Google Sheets source <Icon name="external" size={16} />
          </a>
        </div>
      </section>
    </>
  );
}

export function Methodology({ data }: { data: Experiment }) {
  return (
    <>
      <section className="method-intro">
        <div className="eyebrow">FAIR BY CONSTRUCTION</div>
        <h2>
          Good science starts
          <br />
          before the first prediction.
        </h2>
        <p>
          Data yang sama, split yang tetap, dan evaluasi yang transparan. Setiap
          hasil bisa ditelusuri kembali ke ulasan aslinya.
        </p>
      </section>
      <div className="pipeline">
        {[
          {
            n: "01",
            title: "Prepare",
            text: "Validasi sumber, label, dan pengulangan teks.",
            icon: "database" as const,
          },
          {
            n: "02",
            title: "Experiment",
            text: "Training klasik dan zero-shot inference Gemini.",
            icon: "flask" as const,
          },
          {
            n: "03",
            title: "Evaluate",
            text: "Empat metrik, satu test set, dua confusion matrix.",
            icon: "grid" as const,
          },
          {
            n: "04",
            title: "Recommend",
            text: "Timbang kualitas, latensi, biaya, dan keterbatasan.",
            icon: "spark" as const,
          },
        ].map((step) => (
          <div key={step.n}>
            <div className="pipeline-top">
              <span>{step.n}</span>
              <Icon name={step.icon} />
            </div>
            <h3>{step.title}</h3>
            <p>{step.text}</p>
          </div>
        ))}
      </div>
      <div className="two-columns section-gap">
        <section className="card method-card">
          <span className="small-model-icon classic">
            <Icon name="code" />
          </span>
          <h2>Train locally.</h2>
          <p>TF-IDF + Logistic Regression</p>
          <dl>
            <dt>Feature extraction</dt>
            <dd>Unigram + bigram · sublinear TF</dd>
            <dt>Preprocessing</dt>
            <dd>Lowercase; negasi dipertahankan</dd>
            <dt>Classifier</dt>
            <dd>LogisticRegression · C = 1.0</dd>
            <dt>Training</dt>
            <dd>max_iter = 1000 · seed = 42</dd>
            <dt>Vocabulary</dt>
            <dd>{data.classic.vocabulary_size} fitur dari training saja</dd>
          </dl>
        </section>
        <section className="card method-card">
          <span className="small-model-icon llm">
            <Icon name="spark" />
          </span>
          <h2>Prompt deliberately.</h2>
          <p>{data.llm.config.model}</p>
          <dl>
            <dt>Strategy</dt>
            <dd>Zero-shot · tidak ada label test</dd>
            <dt>Temperature</dt>
            <dd>{data.llm.config.temperature} · mengurangi variasi</dd>
            <dt>Output</dt>
            <dd>JSON · positif / negatif</dd>
            <dt>Token budget</dt>
            <dd>{data.llm.config.max_output_tokens.toLocaleString("id-ID")} total · reasoning {data.llm.config.reasoning_effort || "default"}</dd>
            <dt>Validation</dt>
            <dd>Invalid output ditolak, bukan ditebak</dd>
            <dt>Provider</dt>
            <dd>{data.llm.config.provider} · hasil sukses dicache</dd>
          </dl>
        </section>
      </div>
      <section className="card section-gap prompt-card">
        <SectionTitle
          eyebrow="THE EXACT INSTRUCTION"
          title="A small prompt. A clear contract."
        >
          <Tag tone="neutral">Zero-shot</Tag>
        </SectionTitle>
        <pre>{data.prompt}</pre>
        <p>
          <Icon name="info" size={16} /> Ulasan dikirim sebagai data JSON
          terpisah. API key hanya dibaca oleh Python dari .env.local, tidak
          dikirim ke browser.
        </p>
      </section>
      <section className="card section-gap">
        <SectionTitle title="How to read the metrics" />
        <div className="metric-explanations">
          <div>
            <h3>Accuracy</h3>
            <p>Berapa bagian seluruh ulasan yang diprediksi dengan benar.</p>
          </div>
          <div>
            <h3>Precision</h3>
            <p>
              Seberapa dapat dipercaya prediksi setiap kelas. Macro memberi
              bobot sama pada tiap kelas.
            </p>
          </div>
          <div>
            <h3>Recall</h3>
            <p>
              Seberapa banyak ulasan tiap kelas ditemukan. Recall negatif
              mengukur cakupan keluhan.
            </p>
          </div>
          <div>
            <h3>F1-score</h3>
            <p>
              Keseimbangan precision dan recall per kelas, lalu dirata-ratakan.
              Metrik utama eksperimen.
            </p>
          </div>
        </div>
      </section>
      <div className="method-note">
        <Icon name="info" />
        <p>
          Brief menyebut Gemini AI API langsung. OpenRouter menggunakan model
          Gemini melalui perantara; penerimaan jalur ini mengikuti kebijakan
          mentor. Jalur Google langsung juga tersedia, dengan split dan evaluasi
          yang sama.
        </p>
      </div>
    </>
  );
}

export function Report({ data }: { data: Experiment }) {
  const ready = data.llm.metrics !== null;
  return (
    <>
      <section className="report-hero">
        <div className="eyebrow">TECHNICAL RECOMMENDATION</div>
        <h2>
          {ready ? "Choose with evidence." : "Evidence before a verdict."}
        </h2>
        <p>{data.recommendation}</p>
        <Tag tone={ready ? "green" : "amber"}>
          {ready
            ? "Based on measured results"
            : "Rekomendasi sementara · LLM belum lengkap"}
        </Tag>
      </section>
      <div className="two-columns section-gap">
        <section className="card">
          <SectionTitle title="Quality, without the shortcuts" />
          <div className="report-score-row">
            <span>Classic macro-F1</span>
            <strong>{percent(data.classic.metrics.f1)}</strong>
          </div>
          <div className="report-score-row">
            <span>Gemini macro-F1</span>
            <strong>{percent(data.llm.metrics?.f1)}</strong>
          </div>
          <p className="muted">
            Sensitivitas pada satu kemunculan per teks unik: classic{" "}
            {percent(data.unique_text_sensitivity.classic.f1)}, Gemini{" "}
            {percent(data.unique_text_sensitivity.llm?.f1)}. Hanya{" "}
            {data.split.test_groups} template test independen.
          </p>
        </section>
        <section className="card">
          <SectionTitle title="The cost of understanding" />
          <div className="report-score-row">
            <span>Classic API cost</span>
            <strong>
              $0<small>local compute</small>
            </strong>
          </div>
          <div className="report-score-row">
            <span>Gemini test-set cost</span>
            <strong>
              {data.llm.cost_usd === null
                ? "—"
                : `$${data.llm.cost_usd.toFixed(5)}`}
            </strong>
          </div>
          <p className="muted">
            {data.llm.cost_usd === null
              ? "Biaya belum tersedia, bukan nol. Ditampilkan hanya jika provider melaporkan usage cost."
              : `Ekstrapolasi 1.000 ulasan: $${((data.llm.cost_usd / data.llm.completed) * 1000).toFixed(4)} berdasarkan biaya respons sukses; bukan tarif pasti.`}{" "}
            Infrastruktur dan biaya request gagal tidak termasuk.
          </p>
        </section>
      </div>
      <section className="card section-gap">
        <SectionTitle title="What this experiment can’t tell us" />
        <div className="limitation-list">
          {[
            [
              "01",
              "A small effective sample",
              `${data.dataset.rows} baris berasal dari ${data.dataset.unique_texts} kalimat unik. Test hanya ${data.split.test_groups} template; hasil ini bukan bukti signifikansi statistik atau kesiapan produksi.`,
            ],
            [
              "02",
              "Language beyond the dataset",
              "Sarkasme, campuran sentimen, typo, dan domain shift belum terwakili. Model perlu diuji ulang pada ulasan unik di dunia nyata.",
            ],
            [
              "03",
              "Operational reality",
              "Latensi API mencakup jaringan dan retry. Versi model, routing, rate limit, dan biaya dapat berubah. Probabilitas klasik belum dikalibrasi.",
            ],
          ].map(([n, title, text]) => (
            <div key={n}>
              <span>{n}</span>
              <div>
                <h3>{title}</h3>
                <p>{text}</p>
              </div>
            </div>
          ))}
        </div>
      </section>
      <section className="card section-gap">
        <SectionTitle title="Take the evidence with you" />
        <div className="artifact-grid">
          {[
            {
              name: "Experiment notebook",
              desc: "8 sections · executed outputs",
              file: "experiment_notebook.ipynb",
              icon: "code" as const,
            },
            {
              name: "Full experiment report",
              desc: "Methods, metrics & recommendation",
              file: "experiment_report.md",
              icon: "file" as const,
            },
            {
              name: "Prediction records",
              desc: "Ground truth + both model outputs",
              file: "predictions.csv",
              icon: "rows" as const,
            },
            {
              name: "Comparison figure",
              desc: "Metrics & confusion matrices",
              file: "model_comparison_summary.png",
              icon: "grid" as const,
            },
          ].map((a) => (
            <a
              key={a.file}
              href={`/artifacts/${a.file}`}
              download
              className="artifact-link"
            >
              <span className="round-icon">
                <Icon name={a.icon} />
              </span>
              <div>
                <strong>{a.name}</strong>
                <small>{a.desc}</small>
              </div>
              <Icon name="download" size={17} />
            </a>
          ))}
        </div>
      </section>
      <div className="method-note">
        <Icon name="clock" />
        <p>
          Generated{" "}
          {new Date(data.generated_at)
            .toISOString()
            .replace("T", " · ")
            .slice(0, 22)}{" "}
          UTC. Python {data.environment.python} · scikit-learn{" "}
          {data.environment.scikit_learn}. Semua metrik dibaca dari hasil Python
          yang tersimpan.
        </p>
      </div>
    </>
  );
}
