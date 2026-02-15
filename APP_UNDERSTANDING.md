# App Understanding (ML + Embeddings View)

## What this app does
This is a **Streamlit application** that predicts **per-pax quantity (PP)** and builds downstream:
- Client production plans
- Vendor production plans
- Aggressive vendor plans (for selected clients)

The system is **multi-tenant by client**. Each client gets its own:
- dataset
- feature schema
- trained embedding model artifacts
- business rules for planning

## End-to-end flow
1. User selects a client in the Streamlit UI.
2. The app loads a client-specific `Logic` object that defines features and business rules.
3. If model artifacts are missing, the app trains a TensorFlow embedding model for that client.
4. User enters date + menu items (+ optional non-veg context and special-day metadata).
5. For each item, the app predicts per-pax and total quantity.
6. Planner functions convert predictions into client/vendor/aggressive plans.

## Embeddings architecture
The model in `ml_core.py` is a **dynamic multi-input embedding network**:
- Primary item embedding (`menu_items`)
- Context embedding over up to 10 other menu items (`ctx`, average pooled)
- Additional categorical embeddings based on `encoder_columns` (e.g., `sub_category`, `category`, `weekday`, `day_type`, `holiday_type`, `meal_day`)

Dense head:
- Concatenate all embeddings
- Dense(64, relu) -> Dropout(0.3) -> Dense(32, relu) -> Dense(1)

Prediction output:
- `ideal_pp` regression
- Rounded down to nearest `0.005`
- `total_qty = per_pax * mg`

## Artifact strategy
Artifacts are client-prefixed and cached in-memory:
- `<client>_per_pax_tf_model.keras`
- `<client>_<feature>_encoder.pkl`
- `<client>_item_to_subcat.pkl`
- `<client>_item_to_cat.pkl`
- `<client>_cat_to_subs.pkl`

This enables independent model lifecycles per client.

## Client customization model
`client_logic.py` encapsulates configuration and rules via subclasses of `BaseLogic`.
Each client can override:
- feature schema (`encoder_columns`)
- UI category set (`fixed_categories`)
- vendor MG method (`tekion_2group`, `3group`, `day_based`, or none)
- non-veg behavior, fixed PP, aggressive bump, special-day reductions

Supported client patterns currently include:
- Full embedding + vendor + aggressive plans (e.g., Tekion/Clario)
- Embedding + fixed PP client plan (Odessia)
- Embedding + day-based vendor MG (Stripe)
- Embedding with client-plan-only (Tessolve)
- Non-ML multiplier-only mode (Toasttab)

## Planner logic summary
`planner.py` transforms predicted rows into operational plans:
- `client_plan`: direct PP/qty presentation
- `fixed_pp_client_plan`: static PP map regardless of prediction
- `vendor_plan`: computes vendor MG and ordered quantity per client method
- `aggressive_plan`: additional strategic bumping for priority categories
- `special_day_mg`: holiday/day-type reduction factor

## Data assumptions
Training datasets are Excel files and normalized to lowercase snake-case columns.
Expected columns vary by client logic but generally include:
- `date`, `menu_items`, `sub_category`, `category`, `ideal_pp`
- optional: `day_type`, `holiday_type`, `meal_day`, `meal_type`

## Technical observations (senior ML notes)
- The architecture is practical for tabular categorical regression with menu-context effects.
- Unknown item fallback by sub-category is implemented, which improves robustness.
- Training-on-demand in UI is convenient but should eventually move to offline pipelines for production latency control.
- LabelEncoder is functional, but explicit OOV buckets during training could make unknown handling more principled.
- Artifact namespacing by client is a strong multi-tenant design choice.

## Recommended next production upgrades
1. Offline training jobs + model registry/versioning per client.
2. Add evaluation reports per client/day-type slice (not only RMSE).
3. Introduce feature/data drift checks and input schema validation.
4. Persist prediction logs for feedback loops and re-training triggers.
5. Replace implicit fallback with confidence-aware fallback policy.
6. Add automated tests for planner edge cases and OOV handling.
