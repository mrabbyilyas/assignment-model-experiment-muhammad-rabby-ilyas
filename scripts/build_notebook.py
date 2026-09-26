"""Create and execute the final notebook, retaining the official eight sections."""
from pathlib import Path
import shutil
import sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# AI Model Experiment & Evaluation
**Muhammad Rabby Ilyas · Sentiment Lab**

Eksperimen terukur: TF-IDF + Logistic Regression vs Google Gemini.
Delapan section dari starter resmi dipertahankan. Seluruh output menggunakan data resmi;
tidak ada metrik LLM sintetis. Lihat README untuk laporan lengkap dan referensi."""),
md("""## 1. Problem Statement
Tim produk e-commerce ingin mengklasifikasikan sentimen ulasan untuk memantau kepuasan
dan menemukan keluhan. Eksperimen memilih kandidat pendekatan berdasarkan macro-F1,
accuracy, precision/recall, latensi, effort, dan biaya.

Target adalah `positif` dan `negatif`. Hanya teks ulasan menjadi input; nama produk,
review ID dan label test tidak masuk model. Asumsi: label dataset dapat dipakai sebagai
ground truth; sentimen netral/campuran dipaksa menjadi biner. Data kecil dan berulang
tidak cukup untuk menyatakan kesiapan produksi."""),
md("## 2. Import Library & Load Dataset"),
code("""from pathlib import Path
import sys, json
ROOT = Path.cwd()
if not (ROOT / 'scripts').exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from IPython.display import display, Markdown, Image
from scripts.experiment import (
    load_dataset, split_dataset, create_model, evaluate, llm_config,
    run_experiment, SYSTEM_PROMPT,
)
from scripts.reporting import recommendation
df = load_dataset()
display(df.drop(columns='text_group').head())
display(pd.DataFrame({
    'Jumlah': [len(df), df.text_group.nunique(), df.text_group.duplicated().sum(), df.isna().sum().sum()],
}, index=['Baris', 'Teks unik', 'Pengulangan teks', 'Nilai kosong']))
display(df.sentiment.value_counts().rename('Jumlah').to_frame())"""),
md("""## 3. Menyiapkan Train/Test Split
Dataset mengandung pengulangan kalimat. Split random per baris dapat membocorkan teks
yang sama ke train/test. Karena itu **80:20 pada level teks unik**, stratifikasi label,
seed 42, kemudian seluruh baris tiap grup dikembalikan. Semua data asli tetap dipakai.
Proporsi baris boleh berbeda dari 80:20. ID test yang sama dan urutan yang sama digunakan
untuk kedua model. TF-IDF hanya di-fit pada data training."""),
code("""train, test = split_dataset(df)
assert not set(train.text_group) & set(test.text_group)
assert not set(train.review_id) & set(test.review_id)
assert len(train) + len(test) == len(df)
display(pd.DataFrame({
    'Split': ['Train', 'Test'],
    'Baris': [len(train), len(test)],
    'Teks unik': [train.text_group.nunique(), test.text_group.nunique()],
    'Positif': [sum(train.sentiment == 'positif'), sum(test.sentiment == 'positif')],
    'Negatif': [sum(train.sentiment == 'negatif'), sum(test.sentiment == 'negatif')],
}))
print('Test review IDs:', test.review_id.tolist())"""),
md("""## 4. Pendekatan 1 — Model Klasik (Scikit-learn)
### 4.1 Preprocessing & Feature Extraction
TF-IDF unigram + bigram, lowercase, sublinear term frequency. Negasi dipertahankan,
tanpa stopword removal/stemming agresif. Tidak memakai label test untuk tuning.
### 4.2 Training Model
Logistic Regression: C=1.0, max_iter=1000, random_state=42. Model ringan untuk data kecil."""),
code("""model = create_model()
model.fit(train.review_text, train.sentiment)
print('Vocabulary training:', len(model.named_steps['tfidf'].vocabulary_))
print('Parameters:', model.named_steps['classifier'].get_params())"""),
md("### 4.3 Prediksi pada Test Set"),
code("""classic_predictions = model.predict(test.review_text)
display(pd.DataFrame({
    'review_id': test.review_id,
    'actual': test.sentiment,
    'prediction': classic_predictions,
}).head(10))"""),
md("""## 5. Pendekatan 2 — LLM API (Gemini)
### 5.1 Setup API
Default memakai Google Gemini via OpenRouter sesuai konfigurasi pengguna. Jalur langsung
Gemini AI API tersedia dengan `LLM_PROVIDER=gemini` dan `GEMINI_API_KEY`; jalur ini sesuai
endpoint literal brief. OpenRouter adalah perantara dan penerimaannya perlu mengikuti kebijakan mentor.

Key dibaca dari `.env.local` dan tidak dicetak. `RUN_LIVE_LLM=False` menggunakan cache
tanpa biaya baru. Jika belum ada hasil, status pending ditampilkan secara eksplisit."""),
code("""RUN_LIVE_LLM = False  # Ubah ke True hanya untuk menjalankan API pada test set.
config = llm_config()
display(config)  # Konfigurasi non-rahasia, tidak berisi API key."""),
md("""### 5.2 Merancang Prompt
Zero-shot tanpa contoh test. Input review dienkode sebagai data JSON terpisah dari
system prompt; instruksi di dalam ulasan diabaikan. Label dibatasi ke positif/negatif.
Temperature=0 mengurangi variasi, bukan jaminan determinisme. Gemini 3 memakai reasoning
rendah dengan batas 2.048 token (reasoning + jawaban akhir), yang memberi ruang sebelum
model menghasilkan JSON. Konfigurasi aktual ditampilkan di atas dan dibekukan sebelum
panggilan test pertama. Parser menolak jawaban invalid alih-alih menebak label."""),
code("print(SYSTEM_PROMPT)"),
md("""### 5.3 Menjalankan Prediksi pada Test Set
Fungsi bersama menyimpan output per review, latensi, token, ID respons, konfigurasi,
dan biaya jika dilaporkan. Cache terikat pada model/provider/prompt/parameter/teks/ID.
Timeout/429/5xx mendapat retry terbatas. Kegagalan tidak diperlakukan sebagai kelas;
metrik LLM baru dihitung saat semua prediksi test lengkap."""),
code("""results = run_experiment(with_llm=RUN_LIVE_LLM, provider=config['provider'])
predictions = pd.DataFrame(results['predictions'])
assert predictions.review_id.tolist() == test.review_id.tolist()
print('LLM:', results['llm']['status'], results['llm']['completed'], '/', results['llm']['expected'])
display(predictions[['review_id', 'actual', 'classic', 'llm']].head(12))"""),
md("""## 6. Evaluasi dan Perbandingan
### 6.1 Evaluasi Model Klasik
Semua ringkasan precision/recall/F1 menggunakan **macro average**, zero_division=0.
Baris confusion matrix = aktual; kolom = prediksi; urutan negatif, positif."""),
code("""classic_metrics = results['classic']['metrics']
display(pd.DataFrame(classic_metrics['classification_report']).T)
display(pd.DataFrame(classic_metrics['confusion_matrix'], index=['Aktual negatif','Aktual positif'], columns=['Prediksi negatif','Prediksi positif']))"""),
md("### 6.2 Evaluasi LLM API"),
code("""llm_metrics = results['llm']['metrics']
if llm_metrics is not None:
    display(pd.DataFrame(llm_metrics['classification_report']).T)
    display(pd.DataFrame(llm_metrics['confusion_matrix'], index=['Aktual negatif','Aktual positif'], columns=['Prediksi negatif','Prediksi positif']))
else:
    print('Belum tersedia: lengkapi inference seluruh test set. Tidak ada nilai pengganti/sintetis.')"""),
md("### 6.3 Tabel Perbandingan Ringkasan"),
code("""metrics = ['accuracy', 'precision', 'recall', 'f1']
comparison = pd.DataFrame({
    'TF-IDF + Logistic Regression': {m: classic_metrics[m] for m in metrics},
    'Gemini': {m: llm_metrics[m] if llm_metrics else None for m in metrics},
})
display(comparison.map(lambda value: 'Belum lengkap' if pd.isna(value) else f'{value:.2%}'))
display(Image(filename=str(ROOT / 'documentation/model_comparison_summary.png')))"""),
md("""**Interpretasi metrik.** Accuracy adalah proporsi ulasan yang tepat diprediksi.
Precision macro merata-ratakan keandalan prediksi tiap kelas; recall macro merata-ratakan
cakupan tiap kelas. Recall negatif penting untuk keluhan yang jangan sampai terlewat.
F1 macro menyeimbangkan precision/recall **per kelas** lalu merata-ratakannya; dipakai
sebagai metrik utama karena kedua kelas penting. Elemen off-diagonal confusion matrix
menunjukkan arah kesalahan: keluhan terlewat vs false alarm.

**Sensitivitas pengulangan.** Metrik utama mengikuti semua baris test. Evaluasi tambahan
pada satu kemunculan per teks unik menunjukkan dampak bobot template berulang. Ukuran
sampel independen hanya 8 template; jangan mengklaim signifikansi statistik."""),
code("""display(results['unique_text_sensitivity'])
display(results['diagnostic'])
errors = predictions[(predictions.classic != predictions.actual) | (predictions.llm.notna() & (predictions.llm != predictions.actual))]
display(errors.drop_duplicates('review_text')[['review_text', 'actual', 'classic', 'llm']])
print('Kesalahan di atas teramati; tidak menggunakan contoh buatan.')"""),
md("""## 7. Analisis Trade-off dan Limitation
Model klasik membutuhkan training dan label tetapi inference lokal cepat dan tanpa biaya
API. Gemini tidak butuh training lokal dan dapat memanfaatkan pengetahuan bahasa dari
pretraining, tetapi memerlukan jaringan, kontrol biaya, retry, dan penanganan data ke
pihak luar. Probabilitas Logistic Regression bukan confidence produksi yang terkalibrasi.

Latensi klasik mencakup vectorization; latensi API mencakup jaringan dan retry.
Angka cache adalah waktu request asli, bukan kecepatan membaca file. Biaya API kosong
bukan nol; biaya request gagal belum tentu dilaporkan. Dataset kecil dan repetitif,
hanya satu split, tidak mencakup sarkasme, domain shift, dan mixed sentiment. Tidak ada
klaim kausal dari koefisien maupun jaminan generalisasi dari hasil sempurna."""),
code("""display(pd.DataFrame({
    'Model': ['Classic', 'Gemini'],
    'Mean inference ms/review': [results['classic']['mean_latency_ms'], results['llm']['mean_latency_ms']],
    'P95 inference ms/review': [results['classic']['p95_latency_ms'], results['llm']['p95_latency_ms']],
    'API cost USD': [0, results['llm']['cost_usd']],
}))
if llm_metrics:
    delta = llm_metrics['f1'] - classic_metrics['f1']
    print(f'Selisih macro-F1 Gemini - klasik: {delta * 100:.2f} poin persentase.')
else:
    print('Perbandingan performa akhir menunggu hasil LLM lengkap.')"""),
md("""## 8. Rekomendasi Technical Approach
Rekomendasi berikut dibentuk dari hasil aktual, bukan pilihan yang ditetapkan sebelum
pengujian. Sebelum produksi diperlukan lebih banyak teks unik, audit label, validasi
grup/temporal, pengukuran biaya dan p95 latency pada volume nyata, serta monitoring drift."""),
code("display(Markdown(recommendation(results)))"),
md("""**Artefak audit:** `results/experiment.json`, `results/predictions.csv`,
`results/split_manifest.csv`, `results/llm_cache/`, dan README.
Dashboard Next.js bersifat opsional dan mengambil hasil yang sama.
Referensi sumber dan dokumentasi API lengkap tersedia di README."""),
]


def main():
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    # Force this exact interpreter instead of relying on a globally registered kernel.
    kernel_dir = ROOT / ".venv/share/jupyter/kernels/sentiment-lab"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(__import__('json').dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "Sentiment Lab", "language": "python"}))
    manager = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
    client = NotebookClient(notebook, timeout=600, kernel_name="sentiment-lab", resources={"metadata": {"path": str(ROOT)}})
    client.create_kernel_manager().kernel_spec_manager = manager
    client.execute()
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    target = ROOT / "notebook/experiment_notebook.ipynb"
    nbf.write(notebook, target)
    shutil.copyfile(target, ROOT / "public/artifacts/experiment_notebook.ipynb")
    print(f"Notebook executed: {len(cells)} cells; saved with outputs.")


if __name__ == '__main__':
    main()
