"""Build the Experiment 002 dataset and run Part A (3-way dialect split across layers)
and Part B (2x2 counterfactual lexical shortcut de-confounding + SAE feature clamping)
on google/gemma-3-1b-it with Gemma Scope 2 SAEs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

from weightliftai import feature_trace as ftr

EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "outputs"
EXP_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Construct controlled dataset (Split A: 3-way parallel dialect; Split B: 2x2 counterfactual)
# ---------------------------------------------------------------------------
# Split A: 10 semantic items per emotion x 4 emotions x 3 language/dialect variants = 120 rows
PARALLEL_ITEMS = [
    # SADNESS (10 triplets: en, es_neutral, es_do)
    ("sadness", 1,
     "I have been feeling down all week and I do not really know why.",
     "Me he sentido desanimado toda la semana y no sé realmente por qué.",
     "Llevo la semana entera bajito de ánimo y ni sé decirte por qué."),
    ("sadness", 2,
     "Rainy days leave me feeling low, as if everything around me is gray.",
     "Los días lluviosos me dejan triste, como si todo a mi alrededor fuera gris.",
     "Los días de aguacero me dejan apachurrado, como con un nublado encima."),
    ("sadness", 3,
     "I felt sad when nobody at work remembered my birthday.",
     "Me sentí triste cuando nadie en el trabajo recordó mi cumpleaños.",
     "Me dio una pena grandísima que nadie en el trabajo se acordara de mi cumpleaños."),
    ("sadness", 4,
     "I was disappointed after hearing that I did not get the job.",
     "Me sentí decepcionado al saber que no obtuve el empleo.",
     "Me quedé desencantado cuando me dijeron que no me dieron el puesto."),
    ("sadness", 5,
     "Some afternoons I just feel empty and tired for no clear reason.",
     "Algunas tardes simplemente me siento vacío y cansado sin razón clara.",
     "Hay tardes que me entra un desgano y un vacío raro sin motivo."),
    ("sadness", 6,
     "She looked sad on the bus ride home after her team lost.",
     "Ella se veía triste en el autobús de regreso a casa después de que su equipo perdió.",
     "Ella iba cabizbaja en la guagua pa' su casa después que su equipo perdió."),
    ("sadness", 7,
     "It hurts that my childhood friend and I barely speak anymore.",
     "Me duele que mi amigo de la infancia y yo apenas hablemos ahora.",
     "Me da sentimiento que mi pana de toda la vida y yo ya ni nos hablemos."),
    ("sadness", 8,
     "I cried quietly after finishing the movie by myself.",
     "Lloré en silencio después de terminar de ver la película solo.",
     "Se me salieron las lágrimas calladito cuando se acabó la película."),
    ("sadness", 9,
     "I miss how simple our weekends used to be a few years ago.",
     "Extraño lo sencillos que solían ser nuestros fines de semana hace unos años.",
     "Me hace falta cómo eran los fines de semana antes, me entra una nostalgia."),
    ("sadness", 10,
     "Walking back from the corner store, I felt a quiet wave of loneliness.",
     "Caminando de regreso de la tienda, sentí una silenciosa ola de soledad.",
     "Volviendo del colmado me entró una soledad callada que me dejó sin ánimo."),

    # GRIEF (10 triplets: en, es_neutral, es_do)
    ("grief", 11,
     "Since my father died, I keep setting his plate at the dinner table.",
     "Desde que murió mi padre, sigo poniendo su plato en la mesa para cenar.",
     "Desde que mi papá faltó, sigo poniéndole su plato en la mesa a la hora de comer."),
    ("grief", 12,
     "A year after burying my grandmother, the house feels impossibly quiet.",
     "Un año después de enterrar a mi abuela, la casa se siente increíblemente silenciosa.",
     "Ya va un año del entierro de mi abuela y esta casa se siente vacía de verdad."),
    ("grief", 13,
     "I cannot bring myself to delete the voice messages my mother left me.",
     "No soy capaz de borrar los mensajes de voz que me dejó mi madre.",
     "No tengo corazón para borrar las notas de voz que me dejó mi vieja antes de irse."),
    ("grief", 14,
     "At the funeral I felt numb, and a week later I could not stop weeping.",
     "En el funeral me sentí insensible, y una semana después no podía dejar de llorar.",
     "En el velorio me quedé frío, y a la semana fue que me desmonté a llorar sin consuelo."),
    ("grief", 15,
     "Her son passed away last spring and she folds his clothes every Sunday.",
     "Su hijo falleció la primavera pasada y ella dobla su ropa cada domingo.",
     "Su muchacho partió el año pasado y ella sigue doblándole su ropa todos los domingos."),
    ("grief", 16,
     "Every holiday dinner reminds our family that my brother is gone forever.",
     "Cada cena navideña le recuerda a nuestra familia que mi hermano se fue para siempre.",
     "Cada Nochebuena nos pega el luto de ver la silla vacía de mi hermano."),
    ("grief", 17,
     "After losing the pregnancy, everyone expected me to return to normal quickly.",
     "Después de perder el embarazo, todos esperaban que volviera a la normalidad rápido.",
     "Después de aquella pérdida, la gente quería que yo siguiera como si nada hubiera pasado."),
    ("grief", 18,
     "I dream that my late wife is talking to me in the kitchen, then I wake up alone.",
     "Sueño que mi difunta esposa habla conmigo en la cocina y luego despierto solo.",
     "Sueño clarito que mi difunta esposa me está colando café y cuando despierto me da un dolor en el alma."),
    ("grief", 19,
     "We scattered my uncle's ashes along the coast where he grew up.",
     "Esparcimos las cenizas de mi tío a lo largo de la costa donde creció.",
     "Fuimos a tirar las cenizas de mi tío en la playa que tanto él adoraba."),
    ("grief", 20,
     "Losing my best friend so suddenly left a hole in my life I cannot fill.",
     "Perder a mi mejor amigo tan repentinamente dejó un vacío en mi vida que no puedo llenar.",
     "La partida tan de repente de mi hermano del alma me dejó un hueco que no se llena con nada."),

    # ANXIETY (10 triplets: en, es_neutral, es_do)
    ("anxiety", 21,
     "I keep thinking something bad is about to happen even when everything is fine.",
     "Sigo pensando que algo malo está por suceder incluso cuando todo está bien.",
     "Tengo un presentimiento malo dándome mente aunque todo esté tranquilo."),
    ("anxiety", 22,
     "My thoughts race all night about small tasks I might forget tomorrow.",
     "Mis pensamientos se aceleran toda la noche por pequeñas tareas que podría olvidar mañana.",
     "Me paso la noche entera dándole mente a mil cosas que se me pueden pasar mañana."),
    ("anxiety", 23,
     "Before every routine meeting at work, my chest feels tight and restless.",
     "Antes de cada reunión rutinaria en el trabajo, siento opresión e inquietud en el pecho.",
     "Antes de cualquier reunión normalita se me aprieta el pecho y me entra una desesperación."),
    ("anxiety", 24,
     "I checked the kitchen stove three times and I am still worried I left it on.",
     "Revisé la estufa de la cocina tres veces y todavía me preocupa haberla dejado encendida.",
     "Chequeé la estufa tres veces y sigo con el pendiente de haberla dejado prendida."),
    ("anxiety", 25,
     "What if I fail the exam, lose my scholarship, and ruin my entire future?",
     "¿Qué pasa si repruebo el examen, pierdo mi beca y arruino todo mi futuro?",
     "¿Y si me cuelgan en ese examen, pierdo la beca y se me cae el mundo encima?"),
    ("anxiety", 26,
     "I cannot fall asleep because I keep replaying every conversation from today.",
     "No puedo conciliar el sueño porque sigo repitiendo cada conversación de hoy.",
     "No pego un ojo en la cama dándole vuelta y vuelta a todo lo que dije hoy."),
    ("anxiety", 27,
     "I feel constantly on edge and unable to relax my shoulders.",
     "Me siento constantemente tenso e incapaz de relajar los hombros.",
     "Vivo con los nervios de punta y una intranquilidad que no me deja quieto."),
    ("anxiety", 28,
     "She has not replied in an hour and my mind is already imagining the worst.",
     "Ella no ha respondido en una hora y mi mente ya está imaginando lo peor.",
     "Lleva una hora sin contestarme y ya tengo la cabeza volada pensando lo peor."),
    ("anxiety", 29,
     "Our finances are stable right now, yet I cannot stop worrying about next year.",
     "Nuestras finanzas están estables ahora, pero no puedo dejar de preocuparme por el próximo año.",
     "Ahorita los cuartos alcanzan, pero no dejo de mortificarme pensando en el año que viene."),
    ("anxiety", 30,
     "Waiting for medical test results makes me jittery and short of breath.",
     "Esperar los resultados de los análisis médicos me pone nervioso y con falta de aire.",
     "Esperar esos resultados médicos me tiene al darme un ataque de nervios y con un sofoco."),

    # FEAR (10 triplets: en, es_neutral, es_do)
    ("fear", 31,
     "A large aggressive dog lunged at me on the sidewalk and I froze.",
     "Un perro grande y agresivo se lanzó contra mí en la acera y me quedé inmóvil.",
     "Un perro grandísimo me tiró un tarascazo en la acera y me quedé frío del susto."),
    ("fear", 32,
     "I heard someone forcing the bedroom door handle at 3 a.m. and could not breathe.",
     "Escuché a alguien forzando la manija de la puerta a las tres de la mañana y no pude respirar.",
     "Oí a alguien forcejeando el llavín a las tres de la mañana y se me fue el aire del terror."),
    ("fear", 33,
     "The car skidded across the wet highway toward the guardrail and I screamed.",
     "El automóvil derrapó por la autopista mojada hacia la barandilla y grité.",
     "El carro patinó en la pista mojada directo a la baranda y pegué un grito del espanto."),
    ("fear", 34,
     "When the airplane dropped suddenly in turbulence, I gripped the seat in terror.",
     "Cuando el avión cayó repentinamente por la turbulencia, me aferré al asiento con terror.",
     "Cuando ese avión dio el bajón de golpe, me agarré del asiento muerto del miedo."),
    ("fear", 35,
     "I saw a venomous snake coiled near my shoe and jumped backward.",
     "Vi una serpiente venenosa enrollada cerca de mi zapato y salté hacia atrás.",
     "Vi una culebra pegadita de mi pie y di un brinco pa' atrás con el corazón en la boca."),
    ("fear", 36,
     "The walls started shaking violently during the earthquake and we ran outside.",
     "Las paredes comenzaron a temblar violentamente durante el terremoto y corrimos afuera.",
     "La casa empezó a remecerse durísimo con el temblor y salimos huyendo pa' la calle."),
    ("fear", 37,
     "A stranger followed me down three dark streets at night and I panicked.",
     "Un desconocido me siguió por tres calles oscuras de noche y entré en pánico.",
     "Un tipo raro me cayó atrás por tres calles oscuras y me entró un pánico horrible."),
    ("fear", 38,
     "The smoke alarm shrieked and I saw bright flames climbing the kitchen curtains.",
     "La alarma de humo sonó y vi llamas brillantes subiendo por las cortinas de la cocina.",
     "Sonó la alarma y vi la candela subiendo por la cortina de la cocina."),
    ("fear", 39,
     "I lost sight of my toddler in the crowded terminal and my heart pounded in terror.",
     "Perdí de vista a mi hijo pequeño en la terminal llena de gente y mi corazón latió con terror.",
     "Se me perdió el niño en aquel gentío y sentí que se me salía el corazón del susto."),
    ("fear", 40,
     "The ocean wave pulled me underwater and I could not find the surface.",
     "La ola del mar me arrastró bajo el agua y no podía encontrar la superficie.",
     "El olón me revolcó en el fondo y me vi ahogándome sin hallar por dónde salir."),
]

# Split B: 2x2 Counterfactual Matrix (10 sentences each = 40 rows, English & Spanish mixed 5/5 per cell)
COUNTERFACTUAL_ITEMS = [
    # Cell 1: explicit_grief (True Grief + lexical cues: family / died / funeral / still / todavía)
    ("explicit_grief", "en", "Since my father died last year, I still set his coffee mug on the table every morning."),
    ("explicit_grief", "en", "My grandmother passed away in May and I still cannot delete her voicemails."),
    ("explicit_grief", "en", "After the funeral, my mother still keeps my brother's bedroom exactly as he left it."),
    ("explicit_grief", "en", "It has been two years since we buried my husband, and I still reach for his hand."),
    ("explicit_grief", "en", "Her son died in the accident and she still wears his winter coat."),
    ("explicit_grief", "es", "Desde que murió mi papá el año pasado, todavía le sirvo su taza de café cada mañana."),
    ("explicit_grief", "es", "Mi abuela falleció en mayo y todavía no puedo borrar sus mensajes de voz."),
    ("explicit_grief", "es", "Después del entierro, mi mamá todavía guarda el cuarto de mi hermano intacto."),
    ("explicit_grief", "es", "Ya van dos años desde que enterramos a mi esposo y todavía busco su mano."),
    ("explicit_grief", "es", "Su hijo murió en el accidente y ella todavía se pone su abrigo."),

    # Cell 2: implicit_grief (True Grief/Bereavement, ZERO death/funeral/family-noun/"still"/"todavía" words)
    ("implicit_grief", "en", "I dialed the familiar number out of habit, then remembered nobody would ever answer that phone again."),
    ("implicit_grief", "en", "Buying a single ticket instead of two made the silence in the theater unbearable."),
    ("implicit_grief", "en", "The closet on the left side of the hallway remains untouched, smelling faintly of cedar and old cologne."),
    ("implicit_grief", "en", "Cooking half the recipe feels pointless when the chair across from me stays empty every evening."),
    ("implicit_grief", "en", "Donating the boxed winter sweaters felt like erasing the last physical trace of our years together."),
    ("implicit_grief", "es", "Marqué aquel número por pura costumbre y de golpe recordé que ya nadie me va a contestar jamás."),
    ("implicit_grief", "es", "Comprar un solo boleto en vez de dos hizo que el silencio en la sala fuera insoportable."),
    ("implicit_grief", "es", "El armario del lado izquierdo permanece intacto, con aquel olor suave a cedro y colonia antigua."),
    ("implicit_grief", "es", "Cocinar la mitad de la receta no tiene sentido cuando la silla frente a mí queda vacía cada noche."),
    ("implicit_grief", "es", "Entregar aquellas cajas de ropa se sintió como borrar la última huella de los años que compartimos."),

    # Cell 3: lexical_trap_neutral (Non-grief / zero sadness, packed with family + death/funeral/buried + "still/todavía")
    ("lexical_trap_neutral", "en", "My grandfather is a retired historian who still catalogs nineteenth-century funeral records for the county museum."),
    ("lexical_grief_trap", "en", "My mother works as a probate lawyer and still reviews estate documents after a client has died."),
    ("lexical_trap_neutral", "en", "My brother is an archaeologist who still studies how ancient kings were buried in stone tombs."),
    ("lexical_trap_neutral", "en", "My father's old lawnmower battery finally died yesterday, but the backup charger still works fine."),
    ("lexical_trap_neutral", "en", "My aunt is a florist who still delivers white lilies to the neighborhood funeral home on Tuesdays."),
    ("lexical_trap_neutral", "es", "Mi abuelo es un historiador jubilado que todavía cataloga actas de entierros antiguos para el museo."),
    ("lexical_trap_neutral", "es", "Mi mamá trabaja como abogada sucesoral y todavía revisa testamentos cuando un cliente ha fallecido."),
    ("lexical_trap_neutral", "es", "Mi hermano es arqueólogo y todavía estudia cómo enterraban a los reyes antiguos en tumbas de piedra."),
    ("lexical_trap_neutral", "es", "La batería vieja del taladro de mi papá murió ayer, pero el cargador de repuesto todavía funciona bien."),
    ("lexical_trap_neutral", "es", "Mi tía tiene una floristería y todavía entrega arreglos a la funeraria del barrio todos los martes."),

    # Cell 4: baseline_neutral (Everyday neutral controls)
    ("baseline_neutral", "en", "The public library opens at nine on Saturday mornings and closes at five."),
    ("baseline_neutral", "en", "I organized the project spreadsheets by quarter and backed them up to the drive."),
    ("baseline_neutral", "en", "The recipe requires two cups of flour, warm water, and a pinch of salt."),
    ("baseline_neutral", "en", "The regional train departs from platform four every thirty minutes."),
    ("baseline_neutral", "en", "We replaced the air filter in the hallway unit before summer started."),
    ("baseline_neutral", "es", "La biblioteca pública abre a las nueve los sábados por la mañana y cierra a las cinco."),
    ("baseline_neutral", "es", "Organicé las hojas de cálculo del proyecto por trimestre y las guardé en el disco."),
    ("baseline_neutral", "es", "La receta lleva dos tazas de harina, agua tibia y una pizca de sal."),
    ("baseline_neutral", "es", "El tren regional sale del andén cuatro cada treinta minutos."),
    ("baseline_neutral", "es", "Cambiamos el filtro de aire del pasillo antes de que empezara el verano."),
]

rows_csv = []
for concept, item_id, en_t, es_n_t, es_do_t in PARALLEL_ITEMS:
    rows_csv.append({"split": "dialect_parallel", "item_id": item_id, "concept": concept, "variant": "en", "lang": "en", "text": en_t})
    rows_csv.append({"split": "dialect_parallel", "item_id": item_id, "concept": concept, "variant": "es_neutral", "lang": "es", "text": es_n_t})
    rows_csv.append({"split": "dialect_parallel", "item_id": item_id, "concept": concept, "variant": "es_do", "lang": "es", "text": es_do_t})

for idx, (cond, lang, txt) in enumerate(COUNTERFACTUAL_ITEMS, start=101):
    # Fix typo in one label if present
    cond_clean = "lexical_trap_neutral" if cond == "lexical_grief_trap" else cond
    rows_csv.append({"split": "counterfactual_2x2", "item_id": idx, "concept": cond_clean, "variant": cond_clean, "lang": lang, "text": txt})

df_all = pd.DataFrame(rows_csv)
csv_path = EXP_DIR / "dialect_and_counterfactual_probes_v0.csv"
df_all.to_csv(csv_path, index=False, encoding="utf-8")
dataset_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
print(f"Saved dataset ({len(df_all)} rows) to {csv_path} | SHA256: {dataset_sha256[:16]}...", flush=True)

# ---------------------------------------------------------------------------
# 2. Load Gemma 3 1B and Gemma Scope 2 SAEs across 4 layers (7, 13, 17, 22)
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-3-1b-it"
SAE_RELEASE = "gemma-scope-2-1b-it-res"
LAYERS = [7, 13, 17, 22]
SAE_IDS = {L: f"layer_{L}_width_16k_l0_medium" for L in LAYERS}

t0 = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_ID} on {device}...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device).eval()
model_rev = getattr(model.config, "_commit_hash", None) or "main"

saes = ftr.load_saes(SAE_RELEASE, SAE_IDS, device=device)
print(f"Loaded SAEs for layers { sorted(saes.keys()) }", flush=True)


@torch.no_grad()
def encode_all_layers(texts: list[str], batch: int = 16) -> dict[int, np.ndarray]:
    """Return max-pooled (over non-BOS tokens) SAE feature matrix (N, 16384) for each layer."""
    blocks = ftr.decoder_layers(model)
    out_mats = {L: [] for L in LAYERS}
    for i in range(0, len(texts), batch):
        cache = {}
        hooks = []
        for L in LAYERS:
            def _mk(layer_idx):
                def _h(mod, inp, out):
                    cache[layer_idx] = (out[0] if isinstance(out, tuple) else out).detach().float()
                return _h
            hooks.append(blocks[L].register_forward_hook(_mk(L)))
        try:
            enc = tok(texts[i : i + batch], return_tensors="pt", padding=True).to(device)
            model(**enc)
        finally:
            for h in hooks:
                h.remove()
        cm = ftr.content_mask(enc["attention_mask"])[..., None]
        for L in LAYERS:
            f = saes[L].encode(cache[L]) * cm
            out_mats[L].append(f.max(1).values.cpu())
    return {L: torch.cat(out_mats[L]).numpy() for L in LAYERS}


# ---------------------------------------------------------------------------
# 3. PART A: 3-Way Parallel Dialect Analysis across Layers 7, 13, 17, 22
# ---------------------------------------------------------------------------
df_dial = df_all[df_all.split == "dialect_parallel"].reset_index(drop=True)
print("Encoding 120 parallel dialect sentences across layers 7, 13, 17, 22...", flush=True)
F_dial = encode_all_layers(df_dial.text.tolist())

CONCEPTS = ["sadness", "grief", "anxiety", "fear"]
VARIANTS = ["en", "es_neutral", "es_do"]


def eval_dialect_layer(Fm: np.ndarray) -> dict:
    X_all = np.log1p(Fm)
    active_cols = X_all.max(0) > 0
    X_all = X_all[:, active_cols]
    y_all = df_dial.concept.values
    v_all = df_dial.variant.values

    clf_fn = lambda: make_pipeline(
        SelectKBest(f_classif, k=min(128, X_all.shape[1])),
        LogisticRegression(max_iter=2000, C=1.0, random_state=42),
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    res = {}
    # Within-variant 5-fold CV
    for v in VARIANTS:
        m = v_all == v
        res[f"cv_{v}"] = round(float(cross_val_score(clf_fn(), X_all[m], y_all[m], cv=skf).mean()), 3)

    # Cross-variant zero-shot transfer
    pairs = [
        ("en", "es_neutral"),
        ("en", "es_do"),
        ("es_neutral", "es_do"),
        ("es_do", "es_neutral"),
        ("es_neutral", "en"),
        ("es_do", "en"),
    ]
    for tr, te in pairs:
        c = clf_fn().fit(X_all[v_all == tr], y_all[v_all == tr])
        res[f"{tr}->{te}"] = round(float(c.score(X_all[v_all == te], y_all[v_all == te])), 3)

    #Matched item-level cosine similarity across parallel triplets (average over 40 items)
    # For each item_id 1..40, compute cosine(en, es_neutral), cosine(en, es_do), cosine(es_neutral, es_do)
    cos_en_esn, cos_en_esdo, cos_esn_esdo = [], [], []
    for item_id in range(1, 41):
        v_en = Fm[(df_dial.item_id == item_id) & (df_dial.variant == "en")][0]
        v_esn = Fm[(df_dial.item_id == item_id) & (df_dial.variant == "es_neutral")][0]
        v_esdo = Fm[(df_dial.item_id == item_id) & (df_dial.variant == "es_do")][0]
        n_en = np.linalg.norm(v_en) + 1e-9
        n_esn = np.linalg.norm(v_esn) + 1e-9
        n_esdo = np.linalg.norm(v_esdo) + 1e-9
        cos_en_esn.append(float((v_en @ v_esn) / (n_en * n_esn)))
        cos_en_esdo.append(float((v_en @ v_esdo) / (n_en * n_esdo)))
        cos_esn_esdo.append(float((v_esn @ v_esdo) / (n_esn * n_esdo)))

    res["item_cos_en~es_neutral"] = round(float(np.mean(cos_en_esn)), 3)
    res["item_cos_en~es_do"] = round(float(np.mean(cos_en_esdo)), 3)
    res["item_cos_es_neutral~es_do"] = round(float(np.mean(cos_esn_esdo)), 3)
    return res


dialect_by_layer = {}
for L in LAYERS:
    dialect_by_layer[L] = eval_dialect_layer(F_dial[L])

df_dialect_summary = pd.DataFrame(dialect_by_layer).T
df_dialect_summary.index.name = "layer"
print("\n=== PART A: 3-WAY DIALECT SPLIT ACROSS LAYERS (EN vs ES_NEUTRAL vs ES_DO) ===")
print(df_dialect_summary.to_string())

# ---------------------------------------------------------------------------
# 4. PART B: 2x2 Counterfactual Lexical Shortcut De-confounding & Feature Clamping
# ---------------------------------------------------------------------------
df_cf = df_all[df_all.split == "counterfactual_2x2"].reset_index(drop=True)
print("\nEncoding 40 counterfactual sentences across layers...", flush=True)
F_cf = encode_all_layers(df_cf.text.tolist())
F_cf_13 = F_cf[13]

# Inspect specific Layer-13 features from Experiment 001 on the 4 counterfactual cells:
# 293 ("people one cares for"), 1886 ("death or mortality"), 4284 ("still"), 384, 397, 2697 ("melancholic and sad feelings"), 549, 514
KEY_FEATS = [293, 1886, 4284, 384, 397, 2697, 549, 514]
CONDITIONS = ["explicit_grief", "implicit_grief", "lexical_trap_neutral", "baseline_neutral"]

feat_rows = []
for f_id in KEY_FEATS:
    row = {"feature": f_id}
    for cond in CONDITIONS:
        vals = F_cf_13[df_cf.concept == cond, f_id]
        row[f"{cond}_mean"] = round(float(vals.mean()), 2)
        row[f"{cond}_hit"] = round(float((vals > 0).mean()), 2)
    feat_rows.append(row)
df_feat_cf = pd.DataFrame(feat_rows)
print("\n=== PART B.1: LAYER-13 KEY FEATURE ACTIVATIONS ACROSS 2x2 COUNTERFACTUAL CELLS ===")
print(df_feat_cf.to_string(index=False))

# Discover true Implicit Grief features (features that fire on BOTH explicit_grief AND implicit_grief,
# while NOT firing on lexical_trap_neutral or baseline_neutral)
m_exp = (df_cf.concept == "explicit_grief").values
m_imp = (df_cf.concept == "implicit_grief").values
m_trap = (df_cf.concept == "lexical_trap_neutral").values
m_base = (df_cf.concept == "baseline_neutral").values

true_grief_score = np.minimum(F_cf_13[m_exp].mean(0), F_cf_13[m_imp].mean(0)) - np.maximum(
    F_cf_13[m_trap].mean(0), F_cf_13[m_base].mean(0)
)
top_true_grief_idx = np.argsort(-true_grief_score)[:10]

# Fetch Neuronpedia labels for top true grief features
NP_SOURCE = "gemma-3-1b-it/13-gemmascope-2-res-16k"
LABEL_CACHE = EXP_DIR.parent / "001-gemma-emotion-sae-map" / "outputs" / "neuronpedia_labels.json"
labels = ftr.fetch_labels(NP_SOURCE, [int(x) for x in top_true_grief_idx] + KEY_FEATS, cache_path=LABEL_CACHE)

true_grief_rows = []
for idx in top_true_grief_idx:
    i = int(idx)
    true_grief_rows.append(
        {
            "feature": i,
            "true_grief_score": round(float(true_grief_score[i]), 2),
            "explicit_grief_mean": round(float(F_cf_13[m_exp, i].mean()), 2),
            "implicit_grief_mean": round(float(F_cf_13[m_imp, i].mean()), 2),
            "lexical_trap_mean": round(float(F_cf_13[m_trap, i].mean()), 2),
            "neutral_mean": round(float(F_cf_13[m_base, i].mean()), 2),
            "label": labels.get(i),
        }
    )
df_true_grief = pd.DataFrame(true_grief_rows)
print("\n=== PART B.2: LEXICAL-INVARIANT ('TRUE') GRIEF FEATURES AT LAYER 13 ===")
print(df_true_grief.to_string(index=False))

# Test a Grief Probe trained ONLY on Experiment 001's explicit grief vs neutral:
# How often does it get fooled by `lexical_trap_neutral` vs how well does it detect `implicit_grief`?
# Before and After clamping lexical shortcut features (1886 = death, 4284 = still, 293 = family/people)!
F_dial_13 = F_dial[13]
# Train a binary Grief vs. other-emotion classifier on Split A only (negatives = sadness/anxiety/fear).
# Split B is never seen in training, so its neutral and trap cells are fully held out.
X_train_grief = np.log1p(F_dial_13)
y_train_grief = (df_dial.concept == "grief").astype(int).values

probe_grief = make_pipeline(
    SelectKBest(f_classif, k=64),
    LogisticRegression(max_iter=2000, C=1.0, random_state=42),
)
probe_grief.fit(X_train_grief, y_train_grief)


def eval_counterfactual_probe(F_eval: np.ndarray, clamp_features: list[int] | None = None) -> dict:
    F_mod = F_eval.copy()
    if clamp_features:
        for cf_id in clamp_features:
            F_mod[:, cf_id] = 0.0
    X_ev = np.log1p(F_mod)
    probs = probe_grief.predict_proba(X_ev)[:, 1]
    preds = (probs >= 0.5).astype(int)
    y_true = df_cf.concept.isin(["explicit_grief", "implicit_grief"]).astype(int).values
    out = {
        "overall_auroc": round(float(roc_auc_score(y_true, probs)), 3),
        "explicit_grief_recall": round(float(preds[m_exp].mean()), 3),
        "implicit_grief_recall": round(float(preds[m_imp].mean()), 3),
        "lexical_trap_false_positive_rate": round(float(preds[m_trap].mean()), 3),
        "baseline_neutral_false_positive_rate": round(float(preds[m_base].mean()), 3),
        "explicit_grief_mean_prob": round(float(probs[m_exp].mean()), 3),
        "implicit_grief_mean_prob": round(float(probs[m_imp].mean()), 3),
        "lexical_trap_mean_prob": round(float(probs[m_trap].mean()), 3),
    }
    return out


probe_unclamped = eval_counterfactual_probe(F_cf_13, clamp_features=None)
probe_clamp_lexical = eval_counterfactual_probe(F_cf_13, clamp_features=[1886, 4284, 293])

print("\n=== PART B.3: GRIEF PROBE ON 2x2 COUNTERFACTUALS (UNCLAMPED vs LEXICAL SHORTCUT CLAMPED) ===")
df_probe_cf = pd.DataFrame({"unclamped": probe_unclamped, "clamped_1886_4284_293": probe_clamp_lexical})
print(df_probe_cf.to_string())


# Causal residual-stream feature clamping during generation:
# If we subtract the SAE reconstruction of lexical shortcut features [1886, 4284] on a lexical trap sentence,
# vs clamping true affective features on an implicit grief sentence
@torch.no_grad()
def causal_clamp_logprob_shift(texts: list[str], clamp_feats: list[int], target_word: str = " grief") -> dict:
    """Measure how much zero-clamping specific SAE features in Layer 13's residual stream
    changes the model's log-probability of predicting an empathetic/grief continuation token.
    """
    blocks = ftr.decoder_layers(model)
    target_id = tok.encode(target_word, add_special_tokens=False)[0]

    def get_logprobs(active_clamp: bool) -> np.ndarray:
        lps = []
        for t in texts:
            prompt = f"Read this sentence and name the primary emotion in one word:\n\"{t}\"\nEmotion:"
            ids = tok(prompt, return_tensors="pt").to(device)

            def _hook(mod, inp, out):
                if not active_clamp:
                    return out
                h = out[0] if isinstance(out, tuple) else out
                f = saes[13].encode(h.float())
                f_zero = f.clone()
                for cid in clamp_feats:
                    f_zero[..., cid] = 0.0
                # Subtract only the clamped features' decoder contribution (error-preserving clamp)
                delta = saes[13].decode(f_zero) - saes[13].decode(f)
                delta[:, 0] = 0.0  # leave the BOS position untouched (huge, uninformative norm)
                h_new = h + delta.to(h.dtype)
                return (h_new,) + tuple(out[1:]) if isinstance(out, tuple) else h_new

            handle = blocks[13].register_forward_hook(_hook)
            try:
                logits = model(**ids).logits[0, -1].float()
                logp = torch.log_softmax(logits, dim=-1)[target_id].item()
                lps.append(logp)
            finally:
                handle.remove()
        return np.array(lps)

    base_lp = get_logprobs(False)
    clamp_lp = get_logprobs(True)
    return {
        "mean_logprob_base": round(float(base_lp.mean()), 3),
        "mean_logprob_clamped": round(float(clamp_lp.mean()), 3),
        "mean_delta_logprob": round(float((clamp_lp - base_lp).mean()), 3),
    }


causal_exp_clamp_lex = causal_clamp_logprob_shift(
    df_cf[m_exp].text.tolist(), [1886, 4284, 293], target_word=" grief"
)
causal_trap_clamp_lex = causal_clamp_logprob_shift(
    df_cf[m_trap].text.tolist(), [1886, 4284, 293], target_word=" grief"
)
top3_true_ids = [int(x) for x in top_true_grief_idx[:3]]
causal_imp_clamp_true = causal_clamp_logprob_shift(
    df_cf[m_imp].text.tolist(), top3_true_ids, target_word=" grief"
)

print("\n=== PART B.4: CAUSAL ERROR-PRESERVING CLAMP IN LAYER 13 RESIDUAL STREAM ===")
print("Explicit grief sentences (clamping lexical features 1886, 4284, 293):", causal_exp_clamp_lex)
print("Lexical trap sentences (clamping lexical features 1886, 4284, 293):", causal_trap_clamp_lex)
print(f"Implicit grief sentences (clamping top-3 true grief features {top3_true_ids}):", causal_imp_clamp_true)

elapsed = round(time.time() - t0, 2)
summary_002 = {
    "experiment_id": "002-dialect-and-counterfactuals",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "model_id": MODEL_ID,
    "model_revision": model_rev,
    "sae_release": SAE_RELEASE,
    "layers_evaluated": LAYERS,
    "dataset_path": "experiments/002-dialect-and-counterfactuals/dialect_and_counterfactual_probes_v0.csv",
    "dataset_sha256": dataset_sha256,
    "dataset_rows": len(df_all),
    "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "torch_version": torch.__version__,
    "runtime_seconds": elapsed,
    "part_a_dialect_by_layer": dialect_by_layer,
    "part_b_key_features_2x2": feat_rows,
    "part_b_true_grief_features": true_grief_rows,
    "part_b_probe_clamping": {
        "unclamped": probe_unclamped,
        "clamped_1886_4284_293": probe_clamp_lexical,
    },
    "part_b_causal_logprob_shift": {
        "explicit_grief_clamp_lexical_1886_4284_293": causal_exp_clamp_lex,
        "lexical_trap_clamp_lexical_1886_4284_293": causal_trap_clamp_lex,
        "implicit_grief_clamp_true_top3": causal_imp_clamp_true,
    },
}

(OUT_DIR / "experiment_002_summary.json").write_text(
    json.dumps(summary_002, indent=2, ensure_ascii=False), encoding="utf-8"
)
df_dialect_summary.to_csv(OUT_DIR / "dialect_layer_metrics.csv")
df_feat_cf.to_csv(OUT_DIR / "counterfactual_key_features.csv", index=False)
df_true_grief.to_csv(OUT_DIR / "true_grief_features.csv", index=False)
print(f"\nDONE in {elapsed}s! Saved all outputs to {OUT_DIR}", flush=True)
