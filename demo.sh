#!/usr/bin/env bash
# Four-image demo: one shared threshold, two defective and two healthy inputs.
set -euo pipefail

usage() {
    cat <<'HELP'
Uso: bash demo.sh [opzioni]

Senza argomenti usa l'ultimo modello bottle, fino a 10 immagini test/good
come riferimenti dimostrativi e prova due immagini difettose e due sane.
Le quattro immagini vengono escluse dai riferimenti di calibrazione.
Non esegue training o download. I percorsi relativi partono dalla cartella
del progetto, anche quando lo script viene lanciato da un'altra directory.

  --category NOME       Categoria per cercare il modello (default: bottle)
  --run CARTELLA        Run specifica contenente best.pt e config.json
  --image FILE          Analizza solo questa immagine invece delle quattro prove
  --healthy-dir DIR     Immagini sane di riferimento; ricalcola la soglia
  --reference-limit N   Massimo numero di riferimenti (default: 10; minimo: 2)
  --device DEVICE       auto, cpu, mps o cuda (default: auto)
  --out-dir DIR         Cartella dei risultati (default: demo/<categoria>)
  --recalibrate         Ricalcola anche una soglia gia presente
  -h, --help            Mostra questo aiuto

Esempi:
  bash demo.sh
  bash demo.sh --image data/mvtec/bottle/test/good/000.png --recalibrate
  bash demo.sh --run runs/bottle-<timestamp> --healthy-dir /path/sane --image /path/foto.png

La soglia salvata viene riutilizzata se compatibile con le immagini scelte;
se una di esse era un riferimento, viene ricalcolata escludendola.
Se cambi modello o dispositivo, usa --recalibrate.
HELP
}

fail() { echo "Errore: $*" >&2; exit 2; }

category="bottle"
run_dir=""
image_path=""
healthy_dir=""
out_dir=""
device="auto"
reference_limit="10"
recalibrate=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --recalibrate) recalibrate=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --category|--run|--image|--healthy-dir|--reference-limit|--device|--out-dir)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "Manca il valore di $1"
            case "$1" in
                --category) category="$2" ;;
                --run) run_dir="$2" ;;
                --image) image_path="$2" ;;
                --healthy-dir) healthy_dir="$2"; recalibrate=1 ;;
                --reference-limit) reference_limit="$2"; recalibrate=1 ;;
                --device) device="$2" ;;
                --out-dir) out_dir="$2" ;;
            esac
            shift 2 ;;
        *) fail "Opzione sconosciuta: $1 (usa --help)" ;;
    esac
done

case "$device" in auto|cpu|mps|cuda) ;; *) fail "Dispositivo non valido: $device" ;; esac
[[ "$reference_limit" =~ ^[0-9]+$ ]] || fail "--reference-limit deve essere un intero >= 2"
[[ "$reference_limit" =~ [1-9] ]] || fail "--reference-limit deve essere >= 2"

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
if [[ -x "$project_dir/.venv/bin/python" ]]; then
    demo_python=("$project_dir/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
    demo_python=(uv run python)
else
    fail "Installa uv ed esegui uv sync nella cartella del progetto."
fi

# Discover files before loading a model checkpoint.
if [[ -z "$run_dir" ]]; then
    run_dir="$("${demo_python[@]}" - "$category" <<'PY'
import sys
from pathlib import Path

from diana.data.mvtec import MVTEC_CATEGORIES

category = sys.argv[1]
if category not in MVTEC_CATEGORIES:
    sys.exit(f"Categoria non valida: {category}")
checkpoints = [p for p in Path("runs").glob(f"{category}-*/best.pt") if p.is_file()]
if not checkpoints:
    sys.exit(f"Nessun modello {category} in runs/. Specifica --run oppure esegui prima il training.")
print(max(checkpoints, key=lambda p: (p.stat().st_mtime_ns, str(p))).parent)
PY
    )"
fi
[[ -f "$run_dir/best.pt" ]] || fail "Checkpoint mancante: $run_dir/best.pt"
[[ -f "$run_dir/config.json" ]] || fail "Configurazione mancante: $run_dir/config.json"

category="$("${demo_python[@]}" - "$run_dir/config.json" <<'PY'
import json
import sys
from diana.data.mvtec import MVTEC_CATEGORIES

category = json.load(open(sys.argv[1]))["category"]
if category not in MVTEC_CATEGORIES:
    sys.exit(f"Categoria non valida nel modello: {category}")
print(category)
PY
)"
data_root="$("${demo_python[@]}" - "$run_dir/config.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["data_path"])
PY
)"

if [[ -z "$image_path" ]]; then
    selected_images="$("${demo_python[@]}" - "$data_root" "$category" <<'PY'
import sys
from pathlib import Path

test = Path(sys.argv[1]) / sys.argv[2] / "test"
images = sorted(p for p in test.glob("*/*") if p.is_file()
                and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
defective = [p for p in images if p.parent.name != "good"]
healthy = [p for p in images if p.parent.name == "good"]
if len(defective) < 2 or len(healthy) < 2:
    sys.exit("Servono almeno due immagini difettose e due sane. In alternativa specifica --image /path/foto.png.")
for image in defective[:2] + healthy[:2]:
    print(image)
PY
    )"
    images=()
    while IFS= read -r selected_image; do images+=("$selected_image"); done <<< "$selected_images"
    names=(defective_1 defective_2 healthy_1 healthy_2)
    expected=(difettosa difettosa sana sana)
else
    images=("$image_path")
    names=(prediction)
    expected=(unknown)
fi
for image_path in "${images[@]}"; do
    [[ -f "$image_path" ]] || fail "Immagine non trovata: $image_path"
done
out_dir="${out_dir:-demo/$category}"
calibration="$out_dir/threshold.json"

echo "Modello: $run_dir"
echo "Categoria: $category | dispositivo: $device"
echo "Numero di prove: ${#images[@]}"

# Older single-image calibrations may contain the new healthy test inputs.
# Recalibrate once rather than leaking references into the four-image demo.
if [[ -f "$calibration" && "$recalibrate" == 0 ]]; then
    overlaps="$("${demo_python[@]}" - "$calibration" "${images[@]}" <<'PY'
import json
import sys
from pathlib import Path
from diana.prediction import file_digest

references = {r["sha256"] for r in json.loads(Path(sys.argv[1]).read_text())["references"]}
print(int(any(file_digest(Path(p)) in references for p in sys.argv[2:])))
PY
    )"
    if [[ "$overlaps" == 1 ]]; then
        echo "La vecchia calibrazione contiene immagini delle prove: ricalcolo escludendole."
        recalibrate=1
    fi
fi

if [[ ! -f "$calibration" || "$recalibrate" == 1 ]]; then
    if [[ -z "$healthy_dir" ]]; then
        healthy_dir="$data_root/$category/test/good"
        echo "Calibrazione dimostrativa su test/good: questi riferimenti non sono un test indipendente."
    fi
    [[ -d "$healthy_dir" ]] || fail "Riferimenti sani non trovati: usa --healthy-dir /path/sane"
    echo "Calcolo della soglia su un massimo di $reference_limit immagini sane..."
    exclusion_args=()
    for image_path in "${images[@]}"; do exclusion_args+=(--exclude-image "$image_path"); done
    "${demo_python[@]}" -m diana.calibrate \
        --run "$run_dir" --device "$device" --images "$healthy_dir" \
        --limit "$reference_limit" "${exclusion_args[@]}" --out "$calibration"
else
    echo "Riutilizzo della soglia: $calibration"
fi

summary_args=()
for ((i=0; i<${#images[@]}; i++)); do
    echo "Prova $((i+1))/${#images[@]} | atteso: ${expected[$i]} | ${images[$i]}"
    report="$out_dir/${names[$i]}.json"
    if ! "${demo_python[@]}" -m diana.predict \
        --run "$run_dir" --device "$device" --image "${images[$i]}" \
        --calibration "$calibration" --out "$out_dir/${names[$i]}.png" --report "$report"; then
        echo "Predizione interrotta. Se soglia e impostazioni non coincidono, usa --recalibrate." >&2
        exit 1
    fi
    summary_args+=(--case "${names[$i]}" "${expected[$i]}" "$report")
done

"${demo_python[@]}" -m diana.demo_summary --out-dir "$out_dir" "${summary_args[@]}"
