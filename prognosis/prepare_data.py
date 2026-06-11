import argparse
import csv
from pathlib import Path


def discover_slide_ids(outputs_dir):
    outputs_dir = Path(outputs_dir)
    static_dir = outputs_dir / "static"
    vmisc_dir = outputs_dir / "19plex_feat"
    if not static_dir.is_dir():
        raise FileNotFoundError(f"Missing static feature directory: {static_dir}")
    if not vmisc_dir.is_dir():
        raise FileNotFoundError(f"Missing VMISC feature directory: {vmisc_dir}")

    static_ids = {path.name[: -len("_preds.pt")] for path in static_dir.glob("*_preds.pt")}
    vmisc_ids = {path.name[: -len("_preds.pt")] for path in vmisc_dir.glob("*_preds.pt")}
    slide_ids = sorted(static_ids.intersection(vmisc_ids))
    if not slide_ids:
        raise FileNotFoundError(f"No paired *_preds.pt files found in {static_dir} and {vmisc_dir}")
    return slide_ids


def read_clinical_csv(path):
    if path is None:
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    clinical = {}
    for row in rows:
        slide_id = row.get("slide_id") or row.get("filename") or row.get("case_id") or row.get("patient_id")
        if slide_id:
            clinical[slide_id.replace(".svs", "")] = row
    return clinical


def build_rows(slide_ids, clinical):
    rows = []
    for idx, slide_id in enumerate(slide_ids):
        source = clinical.get(slide_id, {})
        status = source.get("status", "0")
        censorship = source.get("censorship")
        if censorship in (None, ""):
            try:
                censorship = str(1 - int(float(status)))
            except ValueError:
                censorship = "1"

        rows.append(
            {
                "case_id": source.get("case_id") or source.get("patient_id") or slide_id,
                "slide_id": slide_id,
                "survival_months": source.get("survival_months") or source.get("time") or str(12 + idx),
                "censorship": censorship,
                "status": status,
            }
        )
    return rows


def split_rows(rows, val_fraction):
    if len(rows) < 2:
        return rows, rows
    val_count = max(1, round(len(rows) * val_fraction))
    return rows[:-val_count], rows[-val_count:]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "slide_id", "survival_months", "censorship", "status"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Create phase2 train/val manifests from marker inference outputs.")
    parser.add_argument("--outputs-dir", default="marker/outputs", help="Marker inference output directory.")
    parser.add_argument(
        "--output-root",
        default="data/prognosis",
        help="Root directory where train/ and val/ manifest CSVs are written.",
    )
    parser.add_argument(
        "--clinical-csv",
        default=None,
        help="Optional clinical CSV with slide_id or filename plus survival_months/time and censorship/status.",
    )
    parser.add_argument("--name", default="vmisc", help="Manifest basename.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    slide_ids = discover_slide_ids(args.outputs_dir)
    rows = build_rows(slide_ids, read_clinical_csv(args.clinical_csv))
    train_rows, val_rows = split_rows(rows, args.val_fraction)

    output_root = Path(args.output_root)
    train_path = output_root / "train" / f"{args.name}.csv"
    val_path = output_root / "val" / f"{args.name}.csv"
    write_csv(train_path, train_rows)
    write_csv(val_path, val_rows)

    print(f"Wrote {len(train_rows)} train rows to {train_path}")
    print(f"Wrote {len(val_rows)} val rows to {val_path}")
    if args.clinical_csv is None:
        print("No clinical CSV was provided; generated placeholder survival labels for smoke testing only.")


if __name__ == "__main__":
    main()
