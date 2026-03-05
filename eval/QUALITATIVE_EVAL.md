# Qualitative evaluation (angles -> SMPL-X xyz -> short videos)

Questa pipeline è pensata per confrontare qualitativamente GT e predizione con un passaggio **rigido e allineato** alla conversione `angles -> xyz` usata nel progetto.

## Obiettivo

Convertire una sequenza angolare frame-wise in joint 3D xyz tramite layer **SMPL-X neutro** e usare i joint risultanti per creare video brevi di confronto (GT / Pred / Overlap).

## Prerequisiti hard

- `smplx` configurato con:
  - `gender='NEUTRAL'`
  - `use_pca=False`
  - `use_face_contour=True`
- Modelli in `deps/smpl_models`.
- Disponibile `smpl_x.layer["neutral"]` (stessa istanza/mapping usata in training/eval).

## Formato input

- Input conversione: matrice `features` shape `[T, D]`.
- D supportato: `133`, `169`, `172`.
  - `133`: prepend 36 zeri -> `169`
  - `172`: crop prime 169
  - `169`: invariato

## Estrazione da `results.npy`

- Caricare `results.npy` come dict.
- Usare `motion_emb`.
- Se matrice in `[D, T]`, trasporre in `[T, D]`.
- Validare che la dimensione feature finale sia compatibile con `133/169/172`.

## Slicing fisso a 169

Dopo normalizzazione a 169:

- `root_pose = full_169[:, 0:3]`
- `body_pose = full_169[:, 3:66]`
- `lhand_pose = full_169[:, 66:111]`
- `rhand_pose = full_169[:, 111:156]`
- `jaw_pose = full_169[:, 156:159]`
- `expr = full_169[:, 159:169]`

## Parametri addizionali fissi

- `betas`: ripetizione su T del vettore `FIXED_SHAPE` (10D).
- `leye_pose`, `reye_pose`: zeri shape `[T,3]`.

## Forward SMPL-X

Chiamata layer con:

- `betas`
- `body_pose`
- `global_orient=root_pose`
- `right_hand_pose`
- `left_hand_pose`
- `jaw_pose`
- `leye_pose`
- `reye_pose`
- `expression`

## Output xyz finale

- Estrarre joint con `out.joints[:, smpl_x.joint_idx, :]`.
- Shape output: `[T, J, 3]`.
- `smpl_x.joint_idx` **deve essere lo stesso mapping del repo** (allineamento GT/pred).

## Script aggiunto

Esegui il confronto qualitativo con:

```bash
python eval/phoenix_qualitative_eval.py \
  --gt-file /path/to/gt_angles.npy \
  --results-file /path/to/results.npy \
  --output-mp4 /path/to/output.mp4
```

Opzioni utili:

- `--sample-idx 0` selezione sample in output batch.
- `--show-full-body` mostra anche lower body.
- `--input-gt-local-txt /path/to/labels.csv` overlay gloss per frame (`text,length`).
- `--fps 20`.
- `--device cpu|cuda`.

## Prompt pronto per un altro modello

> Prendi gli angoli in `[T,D]`, normalizza `D` a 169 (`133 -> prepend 36 zeri`, `172 -> crop a 169`), fai slicing fisso dei blocchi: root `[0:3]`, body `[3:66]`, lhand `[66:111]`, rhand `[111:156]`, jaw `[156:159]`, expr `[159:169]`; imposta betas fissi (10D) replicati su `T` e eye pose a zero; fai forward del layer SMPL-X neutro (`use_pca=False`) passando quei tensori; estrai joint con `out.joints[:, joint_idx, :]` e usa quello come xyz per frame.
