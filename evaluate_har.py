#!/usr/bin/env python3
"""Evaluate sampled HAR predictions against independently annotated time intervals."""
import argparse, csv, json, math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def load_annotations(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'video', 'start_seconds', 'end_seconds', 'label'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Annotation CSV requires columns: {sorted(required)}')
        rows = []
        for n, row in enumerate(reader, 2):
            if not row['label'].strip():
                raise ValueError(f'Line {n}: missing label')
            start, end = float(row['start_seconds']), float(row['end_seconds'])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError(f'Line {n}: invalid time interval')
            rows.append({'video': row['video'].strip(), 'start': start, 'end': end,
                         'label': row['label'].strip()})
    return rows


def evaluate(report, annotations, clip_duration, ignore_label):
    video = Path(report['input']).name
    relevant = [a for a in annotations if Path(a['video']).name == video]
    if not relevant:
        raise ValueError(f'No annotations for {video}; edit annotations.csv first')
    pairs, skipped, conflicts = [], 0, 0
    for clip in report['clips']:
        start = float(clip['start_seconds'])
        end = start + clip_duration
        # Only evaluate clips entirely inside one annotated segment.
        matches = [a for a in relevant if a['start'] <= start + 1e-6 and end <= a['end'] + 1e-6]
        if len(matches) != 1:
            skipped += 1
            if len(matches) > 1:
                conflicts += 1
            continue
        if matches[0]['label'] == ignore_label:
            skipped += 1
            continue
        pairs.append({'start_seconds': start, 'true': matches[0]['label'], 'predicted': clip['label']})
    if not pairs:
        raise ValueError('No fully annotated clips. Check intervals, video names and labels.')
    classes = sorted({p['true'] for p in pairs} | {p['predicted'] for p in pairs})
    per_class = {}
    for c in classes:
        tp = sum(p['true'] == c and p['predicted'] == c for p in pairs)
        fp = sum(p['true'] != c and p['predicted'] == c for p in pairs)
        fn = sum(p['true'] == c and p['predicted'] != c for p in pairs)
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class[c] = {'support': support, 'precision': round(precision, 6),
                        'recall': round(recall, 6), 'f1': round(f1, 6)}
    n = len(pairs)
    correct = sum(p['true'] == p['predicted'] for p in pairs)
    # Macro-F1 over ground-truth classes; predictions with no ground-truth support
    # still count as false positives for their predicted class.
    true_classes = sorted({p['true'] for p in pairs})
    return {'video': video, 'mode': report['mode'], 'evaluated_clips': n,
            'skipped_clips': skipped, 'overlapping_annotation_conflicts': conflicts,
            'correct_clips': correct, 'accuracy': round(correct / n, 6),
            'macro_f1': round(sum(per_class[c]['f1'] for c in true_classes) / len(true_classes), 6),
            'macro_f1_definition': 'Unweighted mean of per-class F1 over ground-truth classes',
            'per_class': per_class, 'matches': pairs,
            'warning': 'Clip-level metrics on annotated sampled clips only; not a population-wide or benchmark accuracy estimate.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--ignore-label', default='IGNORE')
    args = p.parse_args()
    report = json.loads(args.predictions.read_text(encoding='utf-8'))
    annotations = load_annotations(args.annotations)
    duration = 16.0 / float(report['fps'])
    result = evaluate(report, annotations, duration, args.ignore_label)
    result['source_predictions'] = str(args.predictions)
    result['source_annotations'] = str(args.annotations)
    result['evaluated_at_utc'] = datetime.now(timezone.utc).isoformat()
    out = args.output or Path('results') / f"{Path(report['input']).stem}_{report['mode']}_evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"Evaluated: {result['evaluated_clips']} clips (skipped {result['skipped_clips']})")
    print(f"Accuracy: {result['accuracy']*100:.2f}%")
    print(f"Macro-F1: {result['macro_f1']:.4f}")
    print(f'Saved: {out}')

if __name__ == '__main__':
    main()
