#!/usr/bin/env bash
# run_all.sh – batch-launch nvdiffrec for every prepared mesh folder

MESH_ROOT="data/blender_vault"
ENV_HDR="data/irrmaps/aerodynamics_workshop_2k.hdr"
CFG_DIR="configs"

mkdir -p "$CFG_DIR"          # where JSON configs are saved

iters=1000
tex_res="2048, 2048"
train_res="1024, 1024"
batch=4
lr="0.03, 0.003"
ks_min="0, 0.25, 0"
env_scale=2.0
bg="white"
save_int=100

run_job () {
  mesh_dir="$1"            # folder containing mesh.obj
  tag="$2"                 # baseline | intrinsic
  out=$(basename "$mesh_dir")"_exp_01_learn-light_${tag}"
  cfg="${CFG_DIR}/${out}.json"

  # skip if output dir already exists
  [ -d "$out" ] && { echo "[SKIP] $out exists"; return; }

  cat > "$cfg" <<-JSON
{
  "ref_mesh"        : "${mesh_dir}/mesh.obj",
  "base_mesh"       : "${mesh_dir}/mesh.obj",
  "random_textures" : true,
  "iter"            : ${iters},
  "save_interval"   : ${save_int},
  "texture_res"     : [${tex_res}],
  "train_res"       : [${train_res}],
  "batch"           : ${batch},
  "learning_rate"   : [${lr}],
  "ks_min"          : [${ks_min}],
  "envmap"          : "${ENV_HDR}",
  "env_scale"       : ${env_scale},
  "background"      : "${bg}",
  "validate"        : true,
  "out_dir"         : "${out}",
  "use_intrinsic"   : $( [[ "$tag" == "intrinsic" ]] && echo true || echo false ),
  "intrinsic_lambda": 0.3,
  "display": [
    {
      "relight": "data/irrmaps/studio_small_09_4k.hdr"
    }
  ]
}
JSON

  python train.py --config "$cfg"
}

# loop over every mesh folder that contains mesh.obj
find "$MESH_ROOT" -type f -name mesh.obj | while read -r obj; do
  folder=$(dirname "$obj")
  run_job "$folder" baseline
  run_job "$folder" intrinsic
done
