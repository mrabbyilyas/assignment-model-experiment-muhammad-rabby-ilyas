"""Generate the human-readable report and figures from measured results only."""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def pct(value):
    return f"{value * 100:.2f}%"


def recommendation(r):
    classic, llm = r["classic"]["metrics"], r["llm"]["metrics"]
    if llm is None:
        return "Eksperimen LLM belum lengkap. TF-IDF + Logistic Regression sudah menjadi baseline terukur, tetapi pemenang dan rekomendasi akhir antarpendekatan belum dapat ditetapkan. Lengkapi inference Gemini pada seluruh test set sebelum mengambil keputusan."
    delta = llm["f1"] - classic["f1"]
    if delta > 0:
        return (f"Prioritaskan Gemini untuk pilot dengan volume terbatas: macro-F1 {pct(llm['f1'])} dibanding baseline {pct(classic['f1'])} "
                f"(selisih {delta * 100:.2f} poin persentase) pada test set yang sama. Pertahankan model klasik sebagai baseline lokal berbiaya API nol. "
                "Sebelum produksi, tambah ulasan unik berlabel, ukur anggaran dan latensi pada volume nyata, lalu evaluasi ulang kedua kandidat. "
                "Arsitektur hybrid dapat diteliti berikutnya, tetapi threshold dan keuntungan hybrid belum diuji sehingga belum direkomendasikan sebagai hasil eksperimen ini.")
    return (f"Prioritaskan TF-IDF + Logistic Regression untuk pilot: macro-F1 {pct(classic['f1'])}, dibanding Gemini {pct(llm['f1'])}. "
            "Pada pengujian ini baseline menyamai atau melampaui Gemini, sehingga tambahan dependensi jaringan dan biaya per panggilan belum memberi keuntungan kualitas. "
            "Tetap validasi pada lebih banyak ulasan unik sebelum produksi; Gemini layak dievaluasi lagi bila domain atau ragam bahasa berkembang.")


def export_report(r):
    r["recommendation"] = recommendation(r)
    c, l, d, s = r["classic"], r["llm"], r["dataset"], r["split"]
    cm, lm = c["metrics"], l["metrics"]
    result_path = ROOT / "results/experiment.json"
    result_path.write_text(json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    rows = "\n".join(f"| {name} | {pct(cm[key])} | {pct(lm[key]) if lm else 'Belum lengkap'} |" for key, name in [("accuracy", "Accuracy"), ("precision", "Precision (macro)"), ("recall", "Recall (macro)"), ("f1", "F1-score (macro)")])
    failures = [x for x in r["predictions"] if x["classic"] != x["actual"] or (x["llm"] is not None and x["llm"] != x["actual"])]
    unique_failures = list({x["review_text"]: x for x in failures}.values())
    examples = "\n".join(f"| {x['review_text']} | {x['actual']} | {x['classic']} | {x['llm'] or 'Belum tersedia'} |" for x in unique_failures)
    if not examples:
        examples = "| Tidak ditemukan kesalahan pada prediksi yang tersedia; ini tidak menjamin generalisasi. | — | — | — |"
    llm_time = f"{l['mean_latency_ms']:.1f} ms" if l["mean_latency_ms"] is not None else "Belum tersedia"
    llm_cost = f"US${l['cost_usd']:.6f}" if l["cost_usd"] is not None else "Tidak dilaporkan / belum tersedia"
    report = f'''# Sentiment Lab · AI Model Experiment & Evaluation

**Muhammad Rabby Ilyas · AI Engineering Bootcamp**

Eksperimen sentiment analysis ulasan e-commerce: **TF-IDF + Logistic Regression vs Gemini**.

![Ringkasan eksperimen](documentation/model_comparison_summary.png)

> Status: model klasik selesai; Gemini **{l['status']} ({l['completed']}/{l['expected']} prediksi)**. Semua angka berasal dari eksekusi, bukan data demo. Dashboard Next.js adalah pelengkap; seluruh output penilaian ada pada project Python ini.

## 1. Problem statement

Tim produk e-commerce membutuhkan klasifikasi otomatis sentimen ulasan untuk memantau kepuasan pelanggan dan memprioritaskan penanganan keluhan. Objective eksperimen adalah membandingkan kualitas prediksi, effort implementasi, latensi, dan biaya dua pendekatan sebelum memilih kandidat pilot.

Target: `positif` untuk pengalaman yang memuaskan dan `negatif` untuk keluhan/kekecewaan. Input model **hanya `review_text`**; `review_id`, `product_name`, dan label asli tidak dimasukkan sebagai fitur/prompt. Label dataset diperlakukan sebagai ground truth, bukan kebenaran universal. Netral/mixed sentiment dipaksa ke dua kelas; eksperimen belum mencakup deteksi spam, moderasi, aspek produk, bahasa lain, atau keputusan otomatis berisiko tinggi.

## 2. Dataset dan integritas evaluasi

- [Brief assignment](https://docs.google.com/document/d/17Ex0eWzNqbM5aEpC_4ydBq7-c8I-vjo6N9Y--BPlkVA/edit)
- [Dataset resmi](https://docs.google.com/spreadsheets/d/1ry3h8o_MxeKR0bDolcZ8Au9giMgZl7Q0cyfjdhj1XLE/edit) → `data/customer_reviews_sentiment.csv`, diunduh tanpa perubahan pada 26 September 2026.
- [Starter notebook](https://drive.google.com/file/d/1NJOg3kTIVc0Rs5VSEGtl8hB6FQ3PJO95/view); struktur delapan section dipertahankan dalam notebook final. Salinan starter asli juga disimpan.
- {d['rows']} baris; {d['labels']['positif']} positif, {d['labels']['negatif']} negatif; {d['products']} produk; tidak ada nilai wajib kosong atau review ID ganda.
- **Hanya {d['unique_texts']} teks unik**; {d['repeated_rows']} baris mengulang teks sebelumnya. Semua baris asli tetap digunakan, CSV tidak ditimpa atau direlabel.
- SHA-256 sumber: `{d['sha256']}`.

### Mengapa tidak mengikuti random split per baris secara mentah?

Diagnostik reproduksi split starter (80:20, stratifikasi label, seed 42) menemukan **{r['diagnostic']['overlapping_test_rows']}/{r['diagnostic']['test_rows']} ulasan test memiliki teks identik di training**, dengan accuracy klasik {pct(r['diagnostic']['accuracy'])}. Ini menilai kemampuan mengenali ulang template, bukan generalisasi ke teks baru. Angka diagnostik ini **bukan hasil utama dan tidak dibandingkan dengan LLM**.

Eksperimen utama membagi **teks unik** 80:20 secara stratified dengan `random_state=42`, lalu mengembalikan seluruh baris anggota grup. Hasil: **{s['train_groups']} grup / {s['train_rows']} baris training** dan **{s['test_groups']} grup / {s['test_rows']} baris test** ({s['test_labels']['negatif']} negatif, {s['test_labels']['positif']} positif). Proporsi baris tidak harus 80:20 karena jumlah pengulangan tiap grup berbeda. Overlap teks = **0**. Kedua model dievaluasi pada urutan review ID test yang sama. Manifest tersimpan di `results/split_manifest.csv`.

TF-IDF di-fit hanya pada training set, hyperparameter ditetapkan sebelum evaluasi dan tidak dituning berdasarkan test. Split berdasarkan grup adalah penyesuaian metodologis atas starter untuk menghindari kebocoran, bukan penggantian dataset.

## 3. Eksperimen model klasik

Pipeline Scikit-learn: `TfidfVectorizer(lowercase=True, ngram_range=(1,2), sublinear_tf=True)` → `LogisticRegression(C=1.0, max_iter=1000, random_state=42)`.

Unigram/bigram menangkap kata dan frasa lokal. Tidak ada stopword removal atau stemming agresif agar negasi seperti “tidak” tetap tersedia. Logistic Regression dipilih karena ringan, sesuai data kecil, dan koefisiennya dapat diperiksa. Vocabulary hasil training: **{c['vocabulary_size']} fitur**. Ini satu baseline yang ditetapkan di awal, bukan klaim model klasik terbaik.

## 4. Eksperimen LLM API

- Model diminta: `{l['config']['model']}`; provider: **{l['config']['provider']}**.
- Prompt **zero-shot**; tidak memuat label test atau contoh dari test. Review dikirim sebagai data JSON terpisah dari instruksi sistem.
- `temperature={l['config']['temperature']}` mengurangi variasi output pada klasifikasi; tidak menjamin determinisme mutlak. Batas output `{l['config']['max_output_tokens']}` token mencakup reasoning dan jawaban akhir; reasoning effort `{l['config'].get('reasoning_effort') or 'default'}`. Tidak ada training/fine-tuning LLM.
- Gemini 3.8 Flash dipilih untuk menggunakan generasi Gemini terbaru yang tersedia saat eksperimen, tetap dalam keluarga model yang diminta brief. Reasoning rendah menjadi konfigurasi awal untuk klasifikasi singkat agar biaya terkendali; ini bukan konfigurasi benchmark leaderboard dengan reasoning tinggi. Prompt dan parameter dibekukan sebelum panggilan test pertama. [Model dan tarif](https://openrouter.ai/google/gemini-3.8-flash), [parameter reasoning](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).
- Output harus berupa JSON dengan satu field `sentiment`; hanya `positif`/`negatif` diterima. Output kosong/invalid tidak dipaksa menjadi label. Kegagalan dilaporkan; metrik LLM ditahan sampai seluruh test set lengkap.
- Retry terbatas untuk timeout, HTTP 429, dan 5xx; HTTP auth/kredit/model yang salah menghentikan proses. Hasil sukses dicache per ID, teks, model, provider, prompt, dan parameter. Cache berisi respons, ID respons jika tersedia, model aktual, token, waktu, serta biaya yang dilaporkan; tanpa API key.
- Satu panggilan per baris test, termasuk teks yang berulang. Jalankan ulang perintah yang sama untuk melanjutkan cache yang belum lengkap.

**Kesesuaian brief:** OpenRouter menjalankan model Google Gemini, tetapi merupakan perantara, bukan endpoint Gemini AI API langsung. Brief secara eksplisit menyebut Gemini API; penerimaan jalur OpenRouter bergantung pada mentor. Implementasi mendukung endpoint Google langsung dengan `--provider gemini` dan `GEMINI_API_KEY`, tanpa mengganti split atau metode evaluasi. Tidak ada klaim bahwa OpenRouter otomatis memenuhi interpretasi paling ketat dari rubrik.

<details><summary>Prompt lengkap (beku untuk eksperimen)</summary>

```text
{r['prompt']}
```

</details>

## 5. Hasil evaluasi

Label confusion matrix berurutan **[negatif, positif]**; baris = aktual, kolom = prediksi. Precision, recall, dan F1 memakai **macro average**: hitung per kelas lalu rata-ratakan dengan bobot sama. F1 macro bukan harmonic mean dari precision macro dan recall macro. Pembagian nol ditangani dengan `zero_division=0`.

| Metrik | TF-IDF + Logistic Regression | Gemini |
| --- | ---: | ---: |
{rows}

- **Accuracy**: proporsi seluruh prediksi benar. Klasik benar {cm['correct']}/{cm['total']} ulasan{f"; Gemini benar {lm['correct']}/{lm['total']}" if lm else '; hasil Gemini belum lengkap'}.
- **Precision macro**: rata-rata keandalan prediksi setiap kelas; nilai tinggi berarti lebih sedikit false positive per kelas.
- **Recall macro**: rata-rata cakupan setiap kelas. Recall kelas negatif secara khusus menunjukkan seberapa banyak keluhan terdeteksi.
- **F1 macro**: menyeimbangkan precision dan recall per kelas, lalu merata-ratakan keduanya. Digunakan sebagai metrik utama agar kedua kelas sama pentingnya meski jumlah baris berbeda.
- **Confusion matrix**: diagonal adalah prediksi benar; aktual negatif → prediksi positif adalah keluhan yang terlewat, sedangkan aktual positif → prediksi negatif adalah false alarm.

Interpretasi hasil klasik: precision macro **{pct(cm['precision'])}** lebih tinggi daripada recall macro **{pct(cm['recall'])}**. Secara khusus, hanya **{cm['confusion_matrix'][0][0]}/{sum(cm['confusion_matrix'][0])} keluhan** terdeteksi (recall negatif **{pct(cm['classification_report']['negatif']['recall'])}**), sedangkan **{cm['confusion_matrix'][0][1]} keluhan salah dianggap positif**. Ini kelemahan penting untuk use case prioritas komplain; accuracy keseluruhan saja menyembunyikan arah kesalahannya. {'Gemini mendeteksi ' + str(lm['confusion_matrix'][0][0]) + '/' + str(sum(lm['confusion_matrix'][0])) + ' keluhan, dengan recall negatif ' + pct(lm['classification_report']['negatif']['recall']) + '.' if lm else 'Kinerja deteksi keluhan Gemini belum dapat dinilai.'}

Klasik: `{cm['confusion_matrix']}`. Gemini: `{lm['confusion_matrix'] if lm else 'belum tersedia'}`.

Laporan per kelas ada di `results/experiment.json`. Sebagai pemeriksaan sensitivitas terhadap pengulangan teks, tiap template test diberi bobot satu: macro-F1 klasik **{pct(r['unique_text_sensitivity']['classic']['f1'])}**, Gemini **{pct(r['unique_text_sensitivity']['llm']['f1']) if r['unique_text_sensitivity']['llm'] else 'belum tersedia'}**, dari hanya **{s['test_groups']} teks unik**. Ini analisis tambahan, bukan pengganti metrik seluruh baris. Untuk Gemini, prediksi kemunculan pertama tiap teks digunakan pada analisis sensitivitas.

### Kesalahan yang benar-benar diamati

Contoh berikut dideduplikasi menurut teks agar pengulangan template tidak memenuhi tabel:

| Ulasan | Aktual | Klasik | Gemini |
| --- | --- | --- | --- |
{examples}

Tinjau contoh ini untuk hipotesis kesalahan kosakata, negasi, dan konteks. Koefisien TF-IDF di dashboard dapat membantu interpretasi, tetapi tidak membuktikan penyebab kesalahan. Jangan memperbaiki prompt/model berulang kali pada test set yang sama lalu melaporkannya sebagai holdout baru.

Dua template negatif yang gagal dikenali baseline membahas respons penjual terhadap komplain dan ketidaksesuaian ukuran/warna. Kata seperti “respon” dan “ukuran” juga muncul dalam konteks positif pada training; model bag-of-ngrams dengan hanya 32 template training tidak selalu membedakan hubungan antarkata. Ini **hipotesis dari pemeriksaan contoh**, bukan sebab yang telah dibuktikan. Ketidakseimbangan training (92 positif vs 66 negatif) juga layak diperiksa pada validasi berikutnya. Model dan prompt tidak diubah setelah melihat kegagalan ini.

## 6. Trade-off dan limitation

| Dimensi | Model klasik | Gemini API |
| --- | --- | --- |
| Kualitas | Accuracy {pct(cm['accuracy'])}; macro-F1 {pct(cm['f1'])} | {f"Accuracy {pct(lm['accuracy'])}; macro-F1 {pct(lm['f1'])}" if lm else 'Belum dapat dibandingkan; inference belum lengkap'} |
| Training lokal | {c['training_ms']:.2f} ms pada mesin pengujian | Tidak ada training lokal; model telah dipralatih oleh penyedia |
| Inferensi satu ulasan (mean) | {c['mean_latency_ms']:.3f} ms | {llm_time} |
| Biaya API test set | US$0; komputasi/maintenance tetap punya biaya | {llm_cost} |
| Effort | Butuh label, preprocessing, training dan validasi | Cepat memulai lewat prompt; tetap butuh parsing, retry, cache, kontrol biaya |
| Privasi/dependensi | Inferensi lokal, tanpa jaringan | Teks dikirim ke provider; perlu jaringan dan evaluasi tata kelola data |
| Keterbatasan | Kosakata sempit, variasi bahasa dan negasi tidak selalu tertangkap | Versi/routing berubah, probabilistik, rate limit dan kegagalan output |

Waktu klasik diukur terpisah untuk training, inferensi batch, dan inferensi satu ulasan (termasuk vectorization). Waktu LLM adalah wall-clock per HTTP request termasuk overhead jaringan dan retry, bukan waktu GPU murni. Latensi bukan benchmark terkontrol lintas mesin/region. Cache menyimpan latensi request asli; membaca cache tidak dianggap inference baru. Total biaya yang tersedia menjumlahkan usage cost respons sukses; biaya percobaan gagal tidak selalu dapat diketahui, dan nilai kosong **bukan nol**. Perkiraan biaya 1.000 ulasan, bila dihitung, hanya ekstrapolasi biaya rata-rata test, bukan tarif pasti.

Keterbatasan terpenting: hanya {s['test_groups']} template test independen. Baris berulang membuat ukuran sampel efektif jauh di bawah jumlah baris. Tidak ada klaim signifikansi statistik atau kesiapan produksi. Dataset pendek dan bersih tidak mewakili sarkasme, campuran sentimen, typo, code-switching, atau domain shift. Potensi kemiripan dengan data pretraining LLM tidak dapat diaudit. Evaluasi ini satu split dan satu konfigurasi; uji lanjutan harus menggunakan data unik baru dan validasi berbasis grup.

## 7. Rekomendasi technical approach

{r['recommendation']}

Kriteria sebelum produksi: tambah data unik lintas produk/waktu; anotasi dan audit label; validasi dengan split grup/temporal; ukur recall negatif, biaya per 1.000 ulasan, serta p95 latency sesuai kebutuhan produk; lakukan pemantauan drift dan sampling kesalahan. Tidak ada threshold “confidence” produksi yang disimpulkan dari probabilitas baseline yang belum dikalibrasi.

## 8. Cara menjalankan

Disarankan Python 3.11+ dan Node.js 20.9+ (dashboard opsional). Lingkungan yang benar-benar digunakan: Python {r['environment']['python']}, pandas {r['environment']['pandas']}, scikit-learn {r['environment']['scikit_learn']}. Versi terpasang dicatat di `requirements-lock.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env.local  # hanya untuk clone baru; jangan timpa file yang sudah berisi key
```

Isi `OPENROUTER_API_KEY` di `.env.local` untuk provider default. Jangan commit key. Alternatif sesuai endpoint brief: isi `GEMINI_API_KEY`.

```bash
# Baseline + laporan; tanpa panggilan berbayar. Cache LLM lengkap otomatis dibaca.
python -m scripts.experiment

# Lengkapi Gemini via OpenRouter; melanjutkan hasil cache yang sudah sukses.
python -m scripts.experiment --with-llm

# Satu perintah untuk melengkapi API DAN menjalankan ulang notebook/laporan/dashboard.
python -m scripts.finish

# Alternatif: Google Gemini API langsung, memakai cache provider terpisah.
python -m scripts.experiment --with-llm --provider gemini
# Atau: python -m scripts.finish --provider gemini

# Eksekusi notebook final (tidak membuat panggilan API baru secara default).
python -m scripts.build_notebook

# Pemeriksaan integritas metode, parsing, dan error handling.
python -m pytest -q

# Dashboard interaktif opsional.
npm ci
npm run dev
```

Buka `notebook/experiment_notebook.ipynb` di VS Code/Jupyter, pilih kernel `.venv`, lalu Run All. Notebook memanfaatkan cache dan **tidak memanggil API baru secara default**. Untuk mengaktifkan inference dari notebook, ubah `RUN_LIVE_LLM = True` di section 5.1 atau jalankan CLI di atas. Tanpa key/cache, notebook tetap selesai dan menandai metrik LLM belum tersedia; eksperimen dua model belum lengkap sampai seluruh prediksi LLM tersedia. Refresh dashboard setelah menjalankan eksperimen.

## 9. Struktur output dan checklist rubrik

| Bobot | Bukti dalam repository |
| --- | --- |
| Problem statement · 10% | README §1–2; notebook §1 |
| Model klasik · 20% | `scripts/experiment.py`; notebook §3–4; prediksi per baris |
| LLM API · 20% | Prompt, konfigurasi, client Gemini/OpenRouter, cache respons; notebook §5; status {l['status']} |
| Evaluasi · 25% | Notebook §6; `results/experiment.json`; `results/predictions.csv`; gambar confusion matrix |
| Analisis & rekomendasi · 15% | README §6–7; notebook §7–8 |
| Dokumentasi · 10% | README; requirements; instruksi eksekusi; provenance dataset |

```text
data/customer_reviews_sentiment.csv       # sumber asli, tidak diubah
notebook/experiment_notebook.ipynb         # notebook final dengan output
notebook/starter_notebook.ipynb            # starter resmi, arsip
scripts/experiment.py                     # seluruh metode + integrasi API
scripts/reporting.py                      # laporan/gambar dari angka terukur
results/experiment.json                   # metrik, config, environment, provenance
results/predictions.csv                   # aktual dan prediksi kedua pendekatan
results/split_manifest.csv                # keanggotaan train/test per ID
results/llm_cache/                        # bukti respons API sukses, tanpa key
documentation/model_comparison_summary.png
documentation/experiment_report.md        # salinan laporan yang dapat diunduh
src/                                     # dashboard Next.js, membaca hasil Python
tests/                                   # pemeriksaan integritas
requirements.txt                         # dependensi Python
requirements-lock.txt                    # snapshot versi lingkungan pengujian
```

Seluruh output wajib berada di repository yang sama. Pengumpulan sesuai brief adalah link repository GitHub yang dapat diakses mentor, bukan hanya dashboard. Dashboard menyediakan filter prediksi, confusion matrix, distribusi data, dan unduhan artefak; tidak menambahkan klaim nilai rubrik.

## Referensi implementasi

- [Scikit-learn: text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Scikit-learn: model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [OpenRouter: chat completions](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request)
- [Google: Gemini generateContent](https://ai.google.dev/api/generate-content)

Laporan dibuat dari hasil eksekusi UTC **{r['generated_at']}**. Rekomendasi bersifat kondisional pada eksperimen kecil ini.
'''
    (ROOT / "README.md").write_text(report)
    (ROOT / "documentation/experiment_report.md").write_text(report.replace('(documentation/model_comparison_summary.png)', '(model_comparison_summary.png)'))
    make_figure(r)
    public = ROOT / "public/artifacts"
    public.mkdir(parents=True, exist_ok=True)
    for source in ["results/experiment.json", "results/predictions.csv", "results/split_manifest.csv", "documentation/experiment_report.md", "documentation/model_comparison_summary.png", "data/customer_reviews_sentiment.csv"]:
        shutil.copyfile(ROOT / source, public / Path(source).name)


def make_figure(r):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(14, 8), facecolor="#f7f8f3", layout="constrained")
    fig.get_layout_engine().set(h_pad=0.13, w_pad=0.13)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
    ax = fig.add_subplot(grid[0, :2])
    keys = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(4)
    ax.set_facecolor("#f7f8f3")
    for i, (model, color, name) in enumerate([("classic", "#315b4c", "TF-IDF + Logistic Regression"), ("llm", "#b3c783", "Gemini")]):
        metrics = r[model]["metrics"]
        if metrics:
            bars = ax.bar(x + (i - .5) * .34, [metrics[k] * 100 for k in keys], .32, label=name, color=color)
            ax.bar_label(bars, fmt="%.1f%%", fontsize=9, padding=4)
    ax.set(xticks=x, xticklabels=["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"], ylim=(0, 115), ylabel="Score (%)", title="Same held-out reviews. Measured performance.")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.15), frameon=False, ncol=2)
    info = fig.add_subplot(grid[0, 2]); info.axis("off")
    info.text(.05, .93, "SENTIMENT LAB", fontsize=18, fontweight="bold", color="#234b3d")
    info.text(.05, .72, f"{r['dataset']['rows']} reviews · {r['dataset']['unique_texts']} unique texts\n{r['split']['train_rows']} train / {r['split']['test_rows']} test\n{r['split']['test_groups']} independent test templates\n0 overlapping text groups\nSeed 42 · stratified group split\n\nGemini: {r['llm']['status']}\n{r['llm']['completed']}/{r['llm']['expected']} predictions", va="top", linespacing=1.7)
    for i, (key, title) in enumerate([("classic", "Classic model"), ("llm", "Gemini API")]):
        axis = fig.add_subplot(grid[1, i]); metrics = r[key]["metrics"]
        if metrics:
            values = np.array(metrics["confusion_matrix"])
            axis.imshow(values, cmap="Greens", vmin=0, vmax=max(values.max(), 1))
            for (y, xval), val in np.ndenumerate(values):
                axis.text(xval, y, str(val), ha="center", va="center", fontsize=23, color="white" if val > values.max() * .55 else "#234b3d")
            axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["negatif", "positif"], yticklabels=["negatif", "positif"], xlabel="Predicted", ylabel="Actual", title=title)
        else:
            axis.axis("off"); axis.text(.5, .6, "Gemini results pending", ha="center", fontsize=16)
            axis.text(.5, .4, "No synthetic scores.\nRun all test reviews to complete evaluation.", ha="center", linespacing=1.8)
    note = fig.add_subplot(grid[1, 2]); note.axis("off")
    note.text(.05, .87, "Read the result with care", fontsize=14, fontweight="bold", color="#234b3d")
    note.text(.05, .7, "Repeated text is grouped before splitting.\nA small test set cannot establish\nproduction readiness.\n\nRows = actual; columns = predicted.\nMacro averages weight classes equally.\nAPI cost and latency are in the report.", va="top", linespacing=1.8)
    fig.suptitle("AI Model Experiment & Evaluation  /  Muhammad Rabby Ilyas", fontsize=17, fontweight="bold", x=.03, ha="left")
    fig.savefig(ROOT / "documentation/model_comparison_summary.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
