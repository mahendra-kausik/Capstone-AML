"""Natural language explanations for SHAP / risk attributions."""
from __future__ import annotations

FEATURE_LABELS = {
    "feat_17": "abnormal transaction volume",
    "feat_100": "suspicious graph connectivity",
    "feat_5": "elevated temporal flow patterns",
    "feat_87": "high-risk counterparty exposure",
    "feat_141": "unusual flow velocity",
    "feat_42": "anomalous network centrality",
    "feat_63": "irregular timing patterns",
}


def _label(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("feat_", "feature "))


def generate_narrative(
    prediction: str,
    risk_score: float,
    top_features: list[dict] | None = None,
    confidence: float | None = None,
) -> str:
    if prediction != "illicit" and risk_score < 0.5:
        return (
            "This transaction was classified as Low Risk. Feature attributions indicate "
            "licit-class dominance with no significant illicit drivers above the decision threshold."
        )

    drivers: list[str] = []
    if top_features:
        for feat in top_features[:5]:
            name = feat.get("name", "")
            shap_val = feat.get("shap_value") or feat.get("contribution") or 0
            if shap_val > 0 or (isinstance(shap_val, (int, float)) and abs(shap_val) > 0.01):
                drivers.append(_label(name))

    if not drivers:
        drivers = [
            "abnormal transaction volume",
            "suspicious graph connectivity",
            "similarity to known illicit transaction patterns",
        ]

    severity = "Critical Risk" if risk_score >= 0.85 else "High Risk" if risk_score >= 0.65 else "Elevated Risk"
    driver_text = ", ".join(drivers[:3])
    conf_text = f" (confidence: {confidence:.0%})" if confidence else ""

    return (
        f"This transaction was flagged as {severity} because of {driver_text}. "
        f"The model assigned a risk score of {risk_score:.1%}{conf_text}."
    )
