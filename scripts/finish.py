"""Complete real LLM inference and refresh every submission artifact, including notebook."""
import argparse
import os
from scripts.experiment import run_experiment
from scripts.build_notebook import main as build_notebook

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=['openrouter', 'gemini'])
    args = parser.parse_args()
    if args.provider:
        os.environ['LLM_PROVIDER'] = args.provider
    result = run_experiment(with_llm=True, provider=args.provider)
    build_notebook()
    if result['llm']['status'] != 'complete':
        print('Belum lengkap: periksa key/kredit API, lalu ulangi perintah yang sama. Hasil sukses sudah tersimpan.')
        raise SystemExit(1)
    print('Selesai: hasil Gemini nyata, README, grafik, notebook dengan output, dan dashboard telah diperbarui.')

if __name__ == '__main__':
    main()
