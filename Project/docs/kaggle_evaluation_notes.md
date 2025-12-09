## Kaggle Evaluation Framework Quick Notes

> Goal: Use simple language to describe the core logic of `kaggle_evaluation` package and `demo.py`, helping students participating in Kaggle competitions for the first time understand how online evaluation runs.

---

### 1. Overall Process (Like a Relay Race)

1. The `predict` function we wrote is wrapped into `InferenceServer`.
2. The official `Gateway` written by Kaggle is responsible for sending the test set batch by batch to `predict` in chronological order.
3. `Gateway` collects each batch of predictions and synthesizes the final `submission.parquet`.
4. Data transmission relies on `relay` (using gRPC), like a two-way walkie-talkie.

---

### 2. What `demo.py` Does

- Top reminder: `predict` must be implemented by yourself, and must respond to each batch within 5 minutes.
- The current `predict` returns `0.0` directly, just a placeholder.
- `DefaultInferenceServer(predict)` will register our `predict` as a listener function for the server.
- If the environment variable `KAGGLE_IS_COMPETITION_RERUN` exists, it means it is in the official evaluation environment, and `inference_server.serve()` will be executed to continuously wait for requests.
- During local debugging, `run_local_gateway(...)` will be called to simulate the official test process using public data.

---

### 3. `kaggle_evaluation.default_inference_server`

- **Class:** `DefaultInferenceServer`
- **Function:** Inherits `templates.InferenceServer`, only overrides `_get_gateway_for_test`, returning `DefaultGateway`.
- **Understanding:** This is the "default gateway + server" combination prepared by the official for us to facilitate local self-testing.

---

### 4. `kaggle_evaluation.default_gateway`

- **Class:** `DefaultGateway`
- **Function:** Inherits `templates.Gateway`, mainly customized three things:
  1. `unpack_data_paths`: Find the directory where the competition data is located (default is `test.csv` in the package).
  2. `generate_data_batches`: Read `test.csv` into `polars.DataFrame`, split into multiple batches by `batch_id` (first column), and send to `predict` at once.
  3. `competition_specific_validation`: No extra validation here (`pass`), but custom check logic can be added in subclasses.
- Constructor also sets: prediction column name `prediction`, row ID column name `batch_id`, response time limit 5 minutes.

---

### 5. `core.templates`

- **`Gateway` Abstract Class** (Protagonists are three abstract methods used by `get_all_predictions`):
  - `unpack_data_paths()`: Bind data paths.
  - `generate_data_batches()`: Slice data and attach row IDs.
  - `competition_specific_validation()`: Write competition-specific prediction checks.
- **`InferenceServer` Abstract Class**:
  - Registers the passed function (e.g., `predict`) to the gRPC server during construction.
  - `serve()`: Will block waiting for requests in Kaggle online mode.
  - `run_local_gateway()`: For local testing, warns if timeout exceeds 15 minutes.

---

### 6. `core.base_gateway`

`BaseGateway` is the parent class of `DefaultGateway`, handling all general details:

- **Initialization:** Establish gRPC client (`relay.Client`), save data path, target column name, row ID column name.
- **Timeout Setting:** `set_response_timeout_seconds` controls the response deadline for `predict` (except for the first batch).
- **Main Process `run()`:**
  1. `unpack_data_paths()` sets data directory.
  2. `get_all_predictions()` loops `generate_data_batches()`.
  3. Call `predict` for each batch, do general validation `competition_agnostic_validation()`, then custom validation.
  4. Collect all predictions, write out `submission.parquet`.
  5. If in official environment, also write `result.json` to record success/failure.
- **Key Validation:** `competition_agnostic_validation` ensures prediction row count aligns with row IDs, preventing cheating or format errors.
- **File Sharing:** `share_files` is used to mount large files to the contestant container (symbolic link locally, `mount` online).
- **Error Handling:** Provide clear user prompts via `GatewayRuntimeErrorType` enumeration.

---

### 7. `core.relay`

- Undertakes "Data Transmission + Serialization" tasks.
- **`Client`:**
  - Responsible for establishing gRPC connection with `InferenceServer`.
  - Allows waiting 15 minutes for the first connection to let the server finish starting.
  - Subsequent requests will be limited by `endpoint_deadline_seconds`.
- **`define_server`:** Register function list (e.g., `predict`) as gRPC server.
- **Serialization `_serialize` / `_deserialize`:**
  - Supports common data types like `list`, `tuple`, `dict`, `numpy`, `pandas`, `polars`.
  - Transmission format uses Parquet / Arrow / NPY, ensuring speed and type safety.

---

### 8. Developer Key Points Summary

- `predict` must return quickly and ensure output row count matches row IDs provided by `Gateway`.
- If state needs to be remembered across batches, variables can be saved at the module level (`global`), but pay attention to memory.
- Before running `run_local_gateway` locally, ensure the data path is correct (default points to `/kaggle/input/...`, map or modify locally).
- Once switched to Kaggle online environment, `InferenceServer` will listen continuously until evaluation ends.
- All logs and errors will be fed back to us via `result.json`, `submission.parquet`.

---

> With this information, we can implement a minimum viable version of `predict` from scratch, simulate Kaggle's online test process locally, and then gradually enhance features and models.
