# PostgreSQL Schema

## users
| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | |
| email | VARCHAR UNIQUE | |
| hashed_password | VARCHAR | bcrypt |
| role | ENUM | admin, analyst, viewer |
| is_active | BOOLEAN | |
| created_at | TIMESTAMPTZ | |

## transactions
| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | |
| user_id | UUID FK | uploader |
| batch_id | UUID | nullable |
| tx_id | VARCHAR | external id |
| time_step | INT | optional |
| features | JSONB | 165 floats |
| created_at | TIMESTAMPTZ | |

## predictions
| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | |
| transaction_id | UUID FK | |
| model_name | VARCHAR | static \| evolve |
| risk_score | FLOAT | |
| prediction | VARCHAR | licit \| illicit |
| confidence | FLOAT | |
| prob_licit | FLOAT | |
| prob_illicit | FLOAT | |
| top_features | JSONB | |
| created_at | TIMESTAMPTZ | |

## shap_results
| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | |
| prediction_id | UUID FK | |
| shap_values | JSONB | sparse top-k |
| top_features | JSONB | |
| nsamples | INT | |
| created_at | TIMESTAMPTZ | |

## drift_events
| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | |
| event_type | VARCHAR | shap_tau \| f1_drop \| manual |
| model_name | VARCHAR | |
| window | VARCHAR | e.g. W3→W4 |
| metric_value | FLOAT | |
| threshold | FLOAT | |
| is_alert | BOOLEAN | |
| payload | JSONB | |
| detected_at | TIMESTAMPTZ | |

## reports
| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | |
| user_id | UUID FK | |
| title | VARCHAR | |
| report_type | VARCHAR | batch \| drift \| custom |
| parameters | JSONB | |
| summary | JSONB | |
| created_at | TIMESTAMPTZ | |

## audit_logs
| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | |
| user_id | UUID FK nullable | |
| action | VARCHAR | |
| resource | VARCHAR | |
| detail | JSONB | |
| ip_address | VARCHAR | |
| created_at | TIMESTAMPTZ | |

## batch_jobs
| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | |
| user_id | UUID FK | |
| filename | VARCHAR | |
| status | VARCHAR | pending \| completed \| failed |
| total_rows | INT | |
| high_risk_count | INT | |
| created_at | TIMESTAMPTZ | |
