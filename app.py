"""Local web chat for the saved RHA and RCF strength models."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
DEFAULTS = {"replacement_pct": 10.0, "age_days": 28.0, "ctrl_strength": 40.0}
RCF_NUMERIC = [
    "pozzolanic_reactivity", "d10", "d50", "d90", "mean_particle_um",
    "specific_surface", "bet_surface", "blaine", "specific_gravity",
    "bulk_density", "water_absorption",
]
RCF_CATEGORIES = {
    "activation_method": {"carbonation", "chemical", "mechanical", "multiple", "none", "thermal"},
    "processing_method": {"combined", "crushing", "milling", "other", "thermal"},
    "mineralogy": {"calcite", "mixed", "other", "portlandite", "quartz"},
    "material_family": {"cdw", "fines", "other", "paste", "powder"},
}
RCF_FIELDS = list(RCF_CATEGORIES) + RCF_NUMERIC


@lru_cache(maxsize=2)
def load_model(material: str):
    if material == "RHA":
        from inference import predict_rha

        model, metadata = predict_rha.load()
        return predict_rha, model, metadata

    from inference import predict_rcf

    model, metadata = predict_rcf.load()
    return predict_rcf, model, metadata


def _number(value, field: str, fallback: float | None = None) -> tuple[float | None, bool]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return fallback, fallback is not None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number.")
    return result, False


def predict(payload: dict) -> dict:
    raw_material = str(payload.get("material") or "").strip().upper()
    material = raw_material or "RHA"
    if material not in {"RHA", "RCF"}:
        raise ValueError("Choose either RHA or RCF.")

    raw_inputs = payload.get("inputs") or {}
    if not isinstance(raw_inputs, dict):
        raise ValueError("Inputs must be sent as an object.")

    values: dict[str, float | str | None] = {}
    assumptions: list[str] = []
    if not raw_material:
        assumptions.append("No material was selected; RHA was used.")

    for field, fallback in DEFAULTS.items():
        value, assumed = _number(raw_inputs.get(field), field, fallback)
        values[field] = value
        if assumed:
            assumptions.append(f"{field} defaulted to {fallback:g}.")

    if values["replacement_pct"] is not None and not 0 <= values["replacement_pct"] <= 100:
        raise ValueError("Replacement must be between 0 and 100%.")
    if values["age_days"] is not None and values["age_days"] <= 0:
        raise ValueError("Curing age must be greater than 0 days.")
    if values["ctrl_strength"] is not None and values["ctrl_strength"] <= 0:
        raise ValueError("Control strength must be greater than 0 MPa.")

    if material == "RCF":
        for field in RCF_FIELDS:
            raw_value = raw_inputs.get(field)
            if field in RCF_CATEGORIES:
                normalized = str(raw_value or "").strip().lower()
                if normalized and normalized not in RCF_CATEGORIES[field]:
                    raise ValueError(f"Unsupported {field.replace('_', ' ')} value: {raw_value}.")
                values[field] = normalized or None
            else:
                value, assumed = _number(raw_value, field)
                values[field] = value
                if assumed:
                    assumptions.append(f"{field.replace('_', ' ')} was left unreported.")

        unreported = sum(value is None for field, value in values.items() if field in RCF_FIELDS)
        if unreported:
            assumptions.append(
                f"{unreported} optional RCF material descriptor(s) were not reported; "
                "the model's missing-value handling was used."
            )

    predictor, model, metadata = load_model(material)
    frame = pd.DataFrame([values])
    if material == "RHA":
        features = predictor.prepare(frame)
    else:
        features = predictor.prepare(frame, metadata)
    strength = float(model.predict(features)[0])

    warnings = []
    control_range = predictor.TRAIN_RANGE["ctrl_strength"]
    if not control_range[0] <= values["ctrl_strength"] <= control_range[1]:
        warnings.append(
            f"Control strength is outside this model's training range of "
            f"{control_range[0]:g}-{control_range[1]:g} MPa."
        )
    strength_range = predictor.TRAIN_RANGE["strength"]
    if not strength_range[0] <= strength <= strength_range[1]:
        warnings.append(
            f"Predicted strength is outside the model's training range of "
            f"{strength_range[0]:g}-{strength_range[1]:g} MPa; treat it as extrapolation."
        )
    if material == "RCF" and values["replacement_pct"] == 0:
        warnings.append("The RCF model was trained on blended mixes above 0% replacement.")

    mae = metadata["reported_performance_grouped_holdout"]["mae"]
    return {
        "material": material,
        "model": metadata["name"],
        "strength_mpa": round(strength, 2),
        "expected_mae_mpa": mae,
        "inputs": values,
        "assumptions": assumptions,
        "warnings": warnings,
    }


class ChatHandler(BaseHTTPRequestHandler):
    def _json(self, body: dict, status: int = 200) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            path = "/index.html"
        target = (WEB_DIR / path.lstrip("/")).resolve()
        if WEB_DIR not in target.parents or not target.is_file():
            self.send_error(404)
            return
        content = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 32_768:
                self._json({"error": "Request is too large."}, 413)
                return
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._json(predict(payload))
        except SystemExit as exc:
            message = str(exc)
            if "libomp.dylib" in message or "xgboost" in message.lower():
                message = "RHA needs the macOS OpenMP runtime. Install it with `brew install libomp`, " \
                          "then restart this app."
            self._json({"error": message or "The model could not be loaded."}, 503)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            message = str(exc)
            if "libomp.dylib" in message or "xgboost" in message.lower():
                message = "RHA needs the macOS OpenMP runtime. Install it with `brew install libomp`, " \
                          "then restart this app."
            self._json({"error": message or "Prediction failed while loading the model."}, 500)

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the local RHA/RCF strength chat app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"Strength chat is running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping strength chat.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()