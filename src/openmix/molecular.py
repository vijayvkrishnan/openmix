"""
Molecular property computation — RDKit-based (optional dependency).

Computes properties from SMILES strings. Falls back gracefully if RDKit
is not installed — the rest of OpenMix works without it.
"""

from __future__ import annotations


try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def is_available() -> bool:
    """Check if RDKit is installed and available."""
    return RDKIT_AVAILABLE


def compute_properties(smiles: str) -> dict | None:
    """
    Compute molecular properties from a SMILES string.

    Returns dict with: molecular_weight, log_p, hbd, hba, tpsa,
    rotatable_bonds, aromatic_rings, complexity.
    Returns None if SMILES is invalid or RDKit is not available.
    """
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "molecular_weight": round(Descriptors.ExactMolWt(mol), 2),
        "log_p": round(Crippen.MolLogP(mol), 2),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "complexity": round(Descriptors.BertzCT(mol), 1),
    }


def compute_hlb_griffin(smiles: str) -> float | None:
    """
    Estimate HLB using Griffin's method: HLB = 20 * (MW_hydrophilic / MW_total).

    Identifies hydrophilic groups (OH, COOH, EO, SO3) and estimates their
    contribution to molecular weight.
    Returns None if SMILES is invalid or RDKit is not available.
    """
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    total_mw = Descriptors.ExactMolWt(mol)
    if total_mw == 0:
        return None

    hydrophilic_mw = 0.0

    # -OH groups (not in rings, not phenolic)
    oh_pattern = Chem.MolFromSmarts("[OX2H]")
    if oh_pattern:
        hydrophilic_mw += len(mol.GetSubstructMatches(oh_pattern)) * 17.0

    # -COOH groups
    cooh_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H]")
    if cooh_pattern:
        hydrophilic_mw += len(mol.GetSubstructMatches(cooh_pattern)) * 45.0

    # Ethylene oxide units (-CH2CH2O-)
    eo_pattern = Chem.MolFromSmarts("[CH2][CH2][OX2]")
    if eo_pattern:
        hydrophilic_mw += len(mol.GetSubstructMatches(eo_pattern)) * 44.0

    # Sulfonate groups
    so3_pattern = Chem.MolFromSmarts("[SX4](=O)(=O)[O-,OH]")
    if so3_pattern:
        hydrophilic_mw += len(mol.GetSubstructMatches(so3_pattern)) * 80.0

    if hydrophilic_mw == 0:
        return None

    hlb = 20.0 * (hydrophilic_mw / total_mw)
    return round(min(hlb, 20.0), 1)


def compute_hansen_parameters(smiles: str) -> dict | None:
    """
    Estimate Hansen Solubility Parameters from molecular descriptors.

    Uses Hoftyzer-Van Krevelen approximation. For production use,
    consider HSPiP or a dedicated HSP library.
    Returns dict with {delta_d, delta_p, delta_h} in MPa^0.5.
    """
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    log_p = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    mw = Descriptors.ExactMolWt(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)

    if mw == 0:
        return None

    aromatic_fraction = rdMolDescriptors.CalcNumAromaticRings(mol) * 78.0 / mw
    delta_d = 15.0 + 3.0 * aromatic_fraction + 0.5 * max(log_p, 0)
    delta_p = 2.0 + 0.08 * tpsa
    delta_h = 2.0 + 2.5 * hbd + 1.0 * hba

    return {
        "delta_d": round(delta_d, 1),
        "delta_p": round(delta_p, 1),
        "delta_h": round(delta_h, 1),
    }


# ---------------------------------------------------------------------------
# Functional group detection for mechanism-based interaction prediction
# ---------------------------------------------------------------------------

# SMARTS patterns for reactive functional groups relevant to drug-excipient
# compatibility. Two categories:
#
# Nucleophilic groups (react WITH excipients):
#   primary_amine -> Maillard reaction with reducing sugars
#   secondary_amine -> slower Maillard reaction
#   ester -> hydrolysis catalyzed by MgSt, moisture
#   thiol -> oxidation (by peroxides in PVP, air)
#   phenol -> oxidation, chelation with metals
#   catechol -> strong metal chelation (Fe, Al, Ca)
#   carboxylic_acid -> salt formation with alkaline excipients
#   aldehyde -> gelatin crosslinking, Schiff base formation
#
# Electrophilic groups (self-reactive or react with nucleophilic excipients):
#   michael_acceptor -> conjugate addition with thiols/amines in excipients,
#     self-polymerization, or reaction with gelatin/protein-based excipients
#   epoxide -> ring-opening with nucleophilic excipients, moisture-sensitive
_FUNCTIONAL_GROUP_SMARTS: dict[str, str] = {
    # Nucleophilic reactive groups
    "primary_amine": "[NX3H2][CX4]",              # -NH2 on sp3 carbon (not amide)
    "aromatic_amine": "[NX3H2][cX3]",             # -NH2 on aromatic carbon
    "secondary_amine": "[NX3H1]([CX4])[CX4]",     # -NH- between two sp3 carbons
    "ester": "[CX3](=O)[OX2;!R][CX4]",            # -C(=O)-O-C (excludes ring O)
    "carbamate": "[OX2][CX3](=O)[NX3]",           # -O-C(=O)-N (carbamate, ester variant)
    "thiol": "[SX2H]",                            # -SH
    "phenol": "[OX2H][cX3]",                      # -OH on aromatic carbon
    "catechol": "[OX2H][cX3][cX3][OX2H]",         # two adjacent aromatic -OH
    "carboxylic_acid": "[CX3](=O)[OX2H]",         # -COOH
    "aldehyde": "[CX3H1](=O)",                    # -CHO
    "sulfonamide": "[NX3][SX4](=O)(=O)",           # -N-SO2- (hydrolysis-susceptible)
    # Electrophilic reactive groups
    "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]", # alpha-beta unsaturated carbonyl
    "epoxide": "[OX2r3]",                          # three-membered ring oxygen (oxirane)
}


def detect_functional_groups(smiles: str) -> dict[str, bool]:
    """
    Detect reactive functional groups from a SMILES string.

    Returns a dict of group_name → present (True/False).
    Used for mechanism-based drug-excipient interaction prediction:
    a drug with a primary amine + a reducing sugar excipient = Maillard risk,
    regardless of whether that specific pair is in the knowledge base.

    Returns empty dict if RDKit is not available or SMILES is invalid.
    """
    if not RDKIT_AVAILABLE:
        return {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    result = {}
    for name, smarts in _FUNCTIONAL_GROUP_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            result[name] = False
            continue
        result[name] = mol.HasSubstructMatch(pattern)

    return result


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid. Returns False if RDKit is not available."""
    if not RDKIT_AVAILABLE or not smiles or not isinstance(smiles, str):
        return False
    return Chem.MolFromSmiles(smiles) is not None


def canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical SMILES, or None if invalid/RDKit unavailable."""
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)
