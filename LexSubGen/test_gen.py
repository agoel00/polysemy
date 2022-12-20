import sys
from pathlib import Path

LEXSUBGEN_ROOT = str(Path().resolve().parent)

if LEXSUBGEN_ROOT not in sys.path:
    sys.path.insert(0, LEXSUBGEN_ROOT)

from lexsubgen import SubstituteGenerator

CONFIGS_PATH = Path().resolve() / "configs"

sg = SubstituteGenerator.from_config(
    str(CONFIGS_PATH / "subst_generators" / "lexsub" / "xlnet_embs.jsonnet")
)

s = 'I went to the bank to deposit money'
tgt = 4
subs, w2id = sg.generate_substitutes([s.split()], [tgt])

print(subs)