import { readFile } from "node:fs/promises";
import path from "node:path";
import Dashboard from "@/components/dashboard";
import type { Experiment } from "@/lib/experiment";
export const dynamic = "force-dynamic";
export default async function Home() {
  let data: Experiment | null = null;
  try {
    data = JSON.parse(
      await readFile(
        path.join(process.cwd(), "results/experiment.json"),
        "utf8",
      ),
    );
  } catch {
    // A fresh checkout can show setup instructions before the first experiment.
  }
  if (data) return <Dashboard initialData={data} />;
  return (
    <main className="empty-start">
      <div className="brand-mark">s.</div>
      <h1>Your experiment starts here.</h1>
      <p>
        Hasil eksperimen belum tersedia. Jalankan project Python untuk
        menyiapkan hasil evaluasi.
      </p>
      <code>python -m scripts.experiment</code>
    </main>
  );
}
