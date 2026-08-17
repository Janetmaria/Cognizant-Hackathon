# API Contract

**Owner: Module 8.** This doc is written FIRST, before Module 2, 6, or 7
build their endpoints — everyone else's routes must conform to what's
decided here. Until Module 8 fills this in, treat every section below
as a placeholder, not a spec.

Status: **skeleton only — not yet filled in.**

## Shared field naming

<!-- TODO(Module 8): e.g. always `store_id`, never `store`; snake_case
     everywhere; align with config.PROCESSED_DTYPES keys where the field
     overlaps processed data (see docs/data_schema.md). -->

## Date format

<!-- TODO(Module 8): confirm `YYYY-MM-DD` (ISO 8601, matches
     config.TRAIN_END_DATE / TEST_START_DATE / TEST_END_DATE format) for
     all request/response date fields. -->

## Standard response envelope

<!-- TODO(Module 8): define the shared wrapper every endpoint returns,
     e.g. envelope shape for success vs. error, pagination (if any),
     status/message fields. -->

## Endpoints

### `GET /data/summary`
Owner: Module 2 (`backend/data_routes.py`)
<!-- TODO(Module 8): request params, response shape. -->

### `GET /forecast/{store_id}`
Owner: Module 7 (`backend/main.py`)
<!-- TODO(Module 8): path params, query params, response shape. -->

### `GET /reorder/{store_id}`
Owner: Module 6 (`backend/reorder_routes.py`)
<!-- TODO(Module 8): path params, query params, response shape. -->
