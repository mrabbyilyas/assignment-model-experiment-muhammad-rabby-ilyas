import type { Experiment } from "@/lib/experiment";
import { SectionTitle, Tag } from "./ui";

export default function Features({ data }: { data: Experiment }) {
  const maxWeight = Math.max(
    ...data.features.positif.map((x) => Math.abs(x.weight)),
    ...data.features.negatif.map((x) => Math.abs(x.weight)),
  );
  return (
    <section className="card section-gap">
      <SectionTitle title="The words behind the baseline">
        <Tag tone="neutral">Training coefficients</Tag>
      </SectionTitle>
      <p className="card-subtitle">
        Bobot fitur Logistic Regression. Asosiasi dalam training, bukan bukti
        penyebab kesalahan.
      </p>
      <div className="feature-grid">
        {(["positif", "negatif"] as const).map((label) => (
          <div key={label}>
            <h3>Lebih terkait dengan {label}</h3>
            {data.features[label].slice(0, 6).map((feature) => (
              <div className="feature-row" key={feature.term}>
                <span>{feature.term}</span>
                <div>
                  <i
                    style={{
                      width: `${(Math.abs(feature.weight) / maxWeight) * 100}%`,
                    }}
                  />
                </div>
                <code>{feature.weight.toFixed(2)}</code>
              </div>
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}
