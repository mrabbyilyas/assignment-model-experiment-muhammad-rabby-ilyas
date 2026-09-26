import { readFile } from "node:fs/promises";
import path from "node:path";
export const dynamic = "force-dynamic";
export async function GET() {
  try {
    const data = JSON.parse(
      await readFile(
        path.join(process.cwd(), "results/experiment.json"),
        "utf8",
      ),
    );
    return Response.json(data, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return Response.json(
      {
        error:
          "Hasil belum tersedia. Jalankan eksperimen Python terlebih dahulu.",
      },
      { status: 503 },
    );
  }
}
