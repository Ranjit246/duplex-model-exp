#!/usr/bin/env bash
#
# Download a chunked checkpoint from the Hugging Face hub, reassemble it,
# (re)initialize the base model if needed, and resume DeepSpeed/Accelerate
# training from it.
#
# The tricky part of resuming is finetune.py's consistency check
# (postprocess_args): on resume EVERY arg must equal the saved config.json
# except output_dir / max_train_steps / resume_from_checkpoint. A single
# mismatch (or an absolute path that points at the original training host)
# aborts the run BEFORE logging is set up. This script keeps things consistent
# by (a) rewriting the path-bearing fields in config.json to THIS host and
# (b) reading every consistency-checked value back out of config.json when it
# launches, so the resume command always matches.
#
# Usage:
#   bash resume_from_checkpoint.sh                          # full flow
#   DO_DOWNLOAD=0 bash resume_from_checkpoint.sh            # skip download, then resume
#   DO_RESUME=0   bash resume_from_checkpoint.sh            # prepare only, don't launch
#   STEP=2000     bash resume_from_checkpoint.sh            # resume a different step
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override via environment). The *_PATH defaults are this host's paths;
# config.json is rewritten to match them so the consistency check passes.
# ---------------------------------------------------------------------------
TOKEN="${TOKEN:-}"
REPO="${REPO:-Ranjit/moshiko-kame-hinglish-ft-exp}"
STEP="${STEP:-1500}"

REPO_DIR="${REPO_DIR:-/mnt/LLM-1-TRAIN-SFT-OUT/EXP/duplex-model-exp}"           # repo root; training runs from here
DL_DIR="${DL_DIR:-/mnt/LLM-1-TRAIN-SFT-OUT/EXP/moshiko-kame-hinglish-ft-exp}"   # where hub files are downloaded
OUTPUT_DIR="${OUTPUT_DIR:-output/moshiko-finetuned}"                            # trainer run dir (config.json lives here), relative to REPO_DIR
TRAIN_DATA="${TRAIN_DATA:-${REPO_DIR}/train_text_oracle_a0b1_events-001-of-001.parquet}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_DIR}/ds_configs/zero2-bf16-optim-offload-h100.json}"
MOSHI_LM_REPO="${MOSHI_LM_REPO:-kyutai/moshiko-pytorch-bf16}"                   # base weights for init_moshi_for_ft
NUM_PROCESSES="${NUM_PROCESSES:-1}"

DO_DOWNLOAD="${DO_DOWNLOAD:-1}"
DO_REASSEMBLE="${DO_REASSEMBLE:-1}"
DO_INIT_MODEL="${DO_INIT_MODEL:-1}"
DO_RESUME="${DO_RESUME:-1}"

BASE="https://huggingface.co/${REPO}/resolve/main"
STEP_DIR="step_${STEP}"
CFG="${OUTPUT_DIR}/config.json"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
# Read a scalar (or first element of a list) out of config.json.
cfg() { python3 -c "import json,sys;v=json.load(open(sys.argv[1]))[sys.argv[2]];print(v[0] if isinstance(v,list) else v)" "${CFG}" "$1"; }

cd "${REPO_DIR}"

# ---------------------------------------------------------------------------
# 1. Download checkpoint files with aria2c (multi-connection, resumable).
#    -x16 = 16 connections/server, -s16 = 16 segments/file, -j4 = 4 files at once,
#    --continue resumes partial files, --file-allocation=none = no pre-alloc.
#    In an aria2c input file each URL's options (dir=) go on the next TAB line.
# ---------------------------------------------------------------------------
if [[ "${DO_DOWNLOAD}" == "1" ]]; then
  log "Downloading ${STEP_DIR} from ${REPO} into ${DL_DIR}/ ..."
  DL_LIST="$(mktemp)"
  cat > "${DL_LIST}" <<EOF
${BASE}/${STEP_DIR}/latest
	dir=${STEP_DIR}
${BASE}/${STEP_DIR}/random_states_0.pkl
	dir=${STEP_DIR}
${BASE}/${STEP_DIR}/zero_to_fp32.py
	dir=${STEP_DIR}
${BASE}/${STEP_DIR}/pytorch_model/mp_rank_00_model_states.pt
	dir=${STEP_DIR}/pytorch_model
${BASE}/${STEP_DIR}/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt.part_00
	dir=${STEP_DIR}/pytorch_model
${BASE}/${STEP_DIR}/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt.part_01
	dir=${STEP_DIR}/pytorch_model
${BASE}/config.json
	dir=.
${BASE}/reassemble_checkpoint.sh
	dir=.
EOF
  mkdir -p "${DL_DIR}"
  ( cd "${DL_DIR}" && aria2c -x16 -s16 -j4 --continue=true --file-allocation=none \
      --header="Authorization: Bearer ${TOKEN}" -i "${DL_LIST}" )
  rm -f "${DL_LIST}"
  log "Download complete."
fi

# ---------------------------------------------------------------------------
# 2. Move into the trainer layout (config.json must sit NEXT TO step_<N>) and
#    reassemble the chunked optimizer file (.part_00 + .part_01 -> .pt).
#    Reassembly needs ~94 GB free disk; it cats the parts then deletes them.
# ---------------------------------------------------------------------------
if [[ "${DO_REASSEMBLE}" == "1" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  if [[ -d "${DL_DIR}/${STEP_DIR}" && ! -d "${OUTPUT_DIR}/${STEP_DIR}" ]]; then
    log "Moving ${STEP_DIR} -> ${OUTPUT_DIR}/"
    mv "${DL_DIR}/${STEP_DIR}" "${OUTPUT_DIR}/"
  fi
  if [[ -f "${DL_DIR}/config.json" && ! -f "${CFG}" ]]; then
    log "Moving config.json -> ${OUTPUT_DIR}/"
    mv "${DL_DIR}/config.json" "${CFG}"
  fi
  if compgen -G "${OUTPUT_DIR}/${STEP_DIR}/pytorch_model/*.part_[0-9][0-9]" > /dev/null; then
    log "Reassembling chunked optimizer file ..."
    bash "${DL_DIR}/reassemble_checkpoint.sh" "${OUTPUT_DIR}/${STEP_DIR}"
  else
    log "No .part_* chunks found — already reassembled, skipping."
  fi
fi

# ---------------------------------------------------------------------------
# 2b. Rewrite the path-bearing fields in config.json to THIS host. The hub copy
#     carries absolute paths from the original training machine; if they don't
#     match what we pass at launch, the consistency check aborts the run.
# ---------------------------------------------------------------------------
if [[ -f "${CFG}" ]]; then
  python3 - "${CFG}" "${TRAIN_DATA}" "${DEEPSPEED_CONFIG}" <<'PY'
import json, sys
p, train, ds = sys.argv[1:4]
d = json.load(open(p))
changed = []
if d.get("train_data_files") != [train]:
    changed.append(f"train_data_files -> {train}"); d["train_data_files"] = [train]
if d.get("deepspeed_config_file") != ds:
    changed.append(f"deepspeed_config_file -> {ds}"); d["deepspeed_config_file"] = ds
if changed:
    json.dump(d, open(p, "w"), indent=4)
    print("patched config.json paths:", "; ".join(changed))
else:
    print("config.json paths already current")
PY
fi

# ---------------------------------------------------------------------------
# 3. Initialize the base model if missing. finetune.py builds the model from
#    model_dir BEFORE applying the checkpoint, so resume needs it present.
# ---------------------------------------------------------------------------
if [[ "${DO_INIT_MODEL}" == "1" ]]; then
  MODEL_DIR="$(cfg model_dir)"
  if [[ -f "${MODEL_DIR}/model.safetensors" || -f "${MODEL_DIR}/moshi_lm_kwargs.json" ]]; then
    log "Base model already present at ${MODEL_DIR} — skipping init."
  else
    log "Initializing base model into ${MODEL_DIR} from ${MOSHI_LM_REPO} ..."
    uv run -m tools.init_moshi_for_ft \
      --moshi_lm_repo "${MOSHI_LM_REPO}" \
      --save_dir "${MODEL_DIR}" \
      --model_dtype "$(cfg model_dtype)"
  fi
fi

# ---------------------------------------------------------------------------
# 4. Verify the checkpoint is fully resumable.
# ---------------------------------------------------------------------------
verify() {
  local d="${OUTPUT_DIR}/${STEP_DIR}" ok=1
  for f in latest random_states_0.pkl zero_to_fp32.py \
           pytorch_model/mp_rank_00_model_states.pt \
           pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt; do
    [[ -e "${d}/${f}" ]] && echo "  OK   ${f}" || { echo "  MISS ${f}"; ok=0; }
  done
  if compgen -G "${d}/pytorch_model/*.part_[0-9][0-9]" > /dev/null; then
    echo "  WARN leftover .part_* chunks — reassembly incomplete"; ok=0
  fi
  [[ -f "${CFG}" ]] && echo "  OK   config.json" || { echo "  MISS config.json (must sit next to ${STEP_DIR})"; ok=0; }
  return $(( ok == 1 ? 0 : 1 ))
}
log "Verifying ${OUTPUT_DIR}/${STEP_DIR} ..."
verify || { log "Checkpoint is NOT fully resumable — fix the above before launching."; exit 1; }
log "Checkpoint verified."

# ---------------------------------------------------------------------------
# 5. Resume. Read every consistency-checked value back out of config.json and
#    feed it to the example launcher (which passes RESUME_FROM_CHECKPOINT through
#    to finetune.py). output_dir / max_train_steps / resume_from_checkpoint are
#    the only args allowed to differ from config.json.
#    NOTE: hyperparameters the launcher doesn't expose (learning rates, weight
#    decay, loss weights, oracle_* params, min_length) rely on finetune.py's
#    argparse defaults matching config.json — true for this checkpoint.
# ---------------------------------------------------------------------------
if [[ "${DO_RESUME}" == "1" ]]; then
  log "Launching resume from ${OUTPUT_DIR}/${STEP_DIR} ..."
  TRAIN_DATA_GLOB="$(cfg train_data_files)" \
  DEEPSPEED_CONFIG="$(cfg deepspeed_config_file)" \
  MODEL_DIR="$(cfg model_dir)" \
  MODEL_DTYPE="$(cfg model_dtype)" \
  MAX_LENGTH="$(cfg max_length)" \
  NUM_EPOCHS="$(cfg num_train_epochs)" \
  PER_DEVICE_BATCH_SIZE="$(cfg per_device_train_batch_size)" \
  GRADIENT_ACCUMULATION_STEPS="$(cfg gradient_accumulation_steps)" \
  NUM_WARMUP_STEPS="$(cfg num_warmup_steps)" \
  LOGGING_STEPS="$(cfg logging_steps)" \
  SAVE_STEPS="$(cfg save_steps)" \
  LOG_FILE="$(cfg log_file)" \
  MOSHI_SPEAKERS="$(python3 -c "import json;print(','.join(json.load(open('${CFG}'))['moshi_speakers']))")" \
  USE_ORACLE="$(python3 -c "import json;print(1 if json.load(open('${CFG}'))['use_oracle'] else 0)")" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  RESUME_FROM_CHECKPOINT="${OUTPUT_DIR}/${STEP_DIR}" \
  bash examples/finetune_accelerate.sh
else
  log "DO_RESUME=0 — prepared only. Ready to resume from ${OUTPUT_DIR}/${STEP_DIR}."
fi
