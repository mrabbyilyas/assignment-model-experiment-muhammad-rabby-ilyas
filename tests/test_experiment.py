"""Methodological and failure-mode checks; tests never make real API calls."""
import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from scripts import experiment as ex


def test_original_dataset_and_group_split_are_reproducible():
    df = ex.load_dataset()
    train, test = ex.split_dataset(df)
    train_again, test_again = ex.split_dataset(df)
    assert len(df) == 200 and df.text_group.nunique() == 40
    assert (len(train), len(test)) == (158, 42)
    assert set(train.review_id).isdisjoint(test.review_id)
    assert set(train.text_group).isdisjoint(test.text_group)
    assert set(train.review_id) | set(test.review_id) == set(df.review_id)
    assert set(train.sentiment) == set(test.sentiment) == set(ex.LABELS)
    assert train.equals(train_again) and test.equals(test_again)
    assert df.groupby('text_group').sentiment.nunique().max() == 1


def test_vectorizer_fits_training_only_and_keeps_negation():
    model = ex.create_model().fit(['tidak rusak bagus', 'tidak bagus rusak'], ['positif', 'negatif'])
    vectorizer = model.named_steps['tfidf']
    original = dict(vectorizer.vocabulary_)
    model.predict(['unseentoken exclusiveholdout'])
    assert vectorizer.vocabulary_ == original
    assert 'unseentoken' not in original
    assert 'tidak' in original and 'tidak bagus' in original


def test_macro_metrics_and_confusion_matrix_orientation():
    result = ex.evaluate(['negatif','negatif','positif','positif'], ['negatif','positif','positif','positif'])
    assert result['confusion_matrix'] == [[1,1],[0,2]]
    assert result['accuracy'] == .75
    assert result['precision'] == pytest.approx((1 + 2/3)/2)
    assert result['recall'] == .75
    assert result['f1'] == pytest.approx((2/3 + .8)/2)


@pytest.mark.parametrize('predictions', [['positif'], ['positif', 'neutral'], []])
def test_incomplete_or_invalid_predictions_cannot_be_scored(predictions):
    with pytest.raises(ValueError):
        ex.evaluate(['negatif','positif'], predictions)


@pytest.mark.parametrize('raw', [None, 'positif', '{"sentiment":"netral"}', '{}', '[]', '{"sentiment":null}', '{"sentiment":"positif","extra":1}', '```json\n{"sentiment":"positif"}\n```'])
def test_parser_never_guesses_invalid_labels(raw):
    with pytest.raises((ValueError, TypeError)):
        ex.parse_prediction(raw)


def test_parser_normalizes_valid_label():
    assert ex.parse_prediction(' {"sentiment":" POSITIF "} ') == 'positif'


def test_pending_and_partial_llm_have_no_metrics_then_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, 'ROOT', tmp_path)
    test = pd.DataFrame({'review_id':[10,20], 'review_text':['bagus','rusak'], 'sentiment':['positif','negatif']})
    config = {'provider':'openrouter','model':'google/test','temperature':0,'max_output_tokens':64,'prompt_sha256':'test'}
    summary, _ = ex.run_llm(test, config, False)
    assert summary['metrics'] is None and summary['status'] == 'pending'
    calls = []
    def mock_call(text, config):
        calls.append(text)
        if text == 'rusak' and calls.count('rusak') == 1:
            raise RuntimeError('simulated API failure')
        label = 'positif' if text == 'bagus' else 'negatif'
        return {'prediction':label,'raw_response':json.dumps({'sentiment':label}), 'latency_ms':100,
                'input_tokens':10,'output_tokens':5,'cost_usd':.0001}
    monkeypatch.setattr(ex, 'call_llm', mock_call)
    summary, _ = ex.run_llm(test, config, True)
    assert summary['status'] == 'partial' and summary['completed'] == 1 and summary['metrics'] is None
    summary, records = ex.run_llm(test, config, True)
    assert summary['status'] == 'complete' and summary['metrics']['accuracy'] == 1
    assert calls.count('bagus') == 1 and calls.count('rusak') == 2
    assert list(records) == [10,20]
    assert summary['cost_usd'] == pytest.approx(.0002)
    # Changing a parameter invalidates the cache, not just changing the model.
    new_summary, _ = ex.run_llm(test, {**config, 'temperature':.2}, False)
    assert new_summary['status'] == 'pending'


def test_api_auth_error_is_not_retried_or_echoed(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key-not-real')
    class Response:
        status_code = 401
        ok = False
        text = 'sensitive response: test-key-not-real'
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        return Response()
    monkeypatch.setattr(ex.requests, 'post', post)
    with pytest.raises(RuntimeError, match='HTTP 401') as error:
        ex.call_llm('test review', ex.llm_config('openrouter'))
    assert 'test-key-not-real' not in str(error.value)
    assert len(calls) == 1
    assert 'sentiment' not in json.loads(calls[0]['json']['messages'][1]['content'])


def test_timeout_has_bounded_retries(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key-not-real')
    calls = []
    def post(*args, **kwargs):
        calls.append(1)
        raise requests.Timeout('do not expose request headers')
    monkeypatch.setattr(ex.requests, 'post', post)
    monkeypatch.setattr(ex.time, 'sleep', lambda seconds: None)
    with pytest.raises(RuntimeError, match='3 percobaan'):
        ex.call_llm('test', ex.llm_config('openrouter'))
    assert len(calls) == 3


def test_direct_gemini_contract(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key-not-real')
    monkeypatch.setenv('GEMINI_MODEL', 'gemini-3.8-flash')
    class Response:
        status_code = 200
        ok = True
        def json(self):
            return {'candidates':[{'content':{'parts':[{'thought':True,'text':'ignored reasoning'}, {'text':'{"sentiment":"negatif"}'}]}}],
                    'usageMetadata':{'promptTokenCount':15,'candidatesTokenCount':6,'thoughtsTokenCount':20}, 'modelVersion':'test-gemini'}
    captured = {}
    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()
    monkeypatch.setattr(ex.requests, 'post', post)
    result = ex.call_llm('rusak', ex.llm_config('gemini'))
    assert result['prediction'] == 'negatif' and result['cost_usd'] is None
    assert captured['headers']['x-goog-api-key'] == 'test-key-not-real'
    assert 'test-key-not-real' not in captured['url']
    assert captured['json']['generationConfig']['temperature'] == 0
    assert captured['json']['generationConfig']['thinkingConfig'] == {'thinkingLevel':'low'}
    assert result['output_tokens'] == 26 and result['reasoning_tokens'] == 20


def test_openrouter_reasoning_budget_and_usage_contract(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key-not-real')
    monkeypatch.setenv('OPENROUTER_MODEL', 'google/gemini-3.8-flash')
    class Response:
        status_code = 200
        ok = True
        def json(self):
            return {'choices':[{'message':{'content':'{"sentiment":"positif"}'}}],
                    'usage':{'prompt_tokens':120,'completion_tokens':56,'cost':.0003,
                             'completion_tokens_details':{'reasoning_tokens':50}},
                    'id':'mock-response','model':'google/gemini-3.8-flash'}
    captured = {}
    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()
    monkeypatch.setattr(ex.requests, 'post', post)
    result = ex.call_llm('sangat bagus', ex.llm_config('openrouter'))
    assert captured['json']['max_tokens'] == 2048
    assert captured['json']['reasoning'] == {'effort':'low','exclude':True}
    assert captured['json']['provider']['allow_fallbacks'] is False
    assert json.loads(captured['json']['messages'][1]['content']) == {'review_text':'sangat bagus'}
    assert result['output_tokens'] == 56 and result['reasoning_tokens'] == 50
    assert result['cost_usd'] == .0003 and result['prediction'] == 'positif'


def test_published_artifacts_match_measured_results():
    result = json.loads((ex.ROOT / 'results/experiment.json').read_text())
    predictions = pd.read_csv(ex.ROOT / 'results/predictions.csv')
    assert predictions.review_id.tolist() == result['split']['test_ids']
    recomputed = ex.evaluate(predictions.actual.tolist(), predictions.classic.tolist())
    assert recomputed == result['classic']['metrics']
    if result['llm']['status'] == 'complete':
        assert ex.evaluate(predictions.actual.tolist(), predictions.llm.tolist()) == result['llm']['metrics']
    else:
        assert result['llm']['metrics'] is None
    assert json.loads((ex.ROOT / 'public/artifacts/experiment.json').read_text()) == result
    assert ex.digest((ex.ROOT / 'data/customer_reviews_sentiment.csv').read_bytes()) == result['dataset']['sha256']
