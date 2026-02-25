#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# === CONFIG & SETUP ==========================================
# === Helper: Lookup taxonomy presentation document for target filing ===

import threading
from lxml import etree
from io import BytesIO
from .http_client import get as http_get

_NEGATED_LABELS_CACHE = {}
_CONCEPT_ROLES_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _cache_key(cik, accession_number):
    return f"{int(cik)}::{accession_number}"


def _clone_concept_roles(concept_roles):
    # Defensive shallow clone so callers can't mutate cached structure.
    return {tag: list(roles) for tag, roles in concept_roles.items()}


def get_negated_label_concepts(cik, accession_number, headers):
    """
    For a given CIK and accession number, fetch the .pre.xml presentation file and return a set of concept names
    (e.g., 'us-gaap:PaymentsToAcquirePropertyPlantAndEquipment') that use a negatedLabel.
    """
    acc_nodash = accession_number.replace("-", "")
    base_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/"
    index_url = base_url + "index.json"
    key = _cache_key(cik, accession_number)

    with _CACHE_LOCK:
        cached = _NEGATED_LABELS_CACHE.get(key)
    if cached is not None:
        return set(cached)

    try:
        r_index = http_get(index_url, headers=headers, rate_limited=True)
        r_index.raise_for_status()
        index_data = r_index.json()
        items = index_data.get("directory", {}).get("item", [])
        pre_file = next((item["name"] for item in items if "pre" in item["name"].lower() and item["name"].endswith(".xml")), None)
        if not pre_file:
            print(f"⚠️ No .pre.xml found for {accession_number}")
            return set()
        
        pre_url = base_url + pre_file
        print(f"🔗 Downloading .pre.xml from: {pre_url}")  # 👈 Add this here
        r_pre = http_get(pre_url, headers=headers, rate_limited=True)
        r_pre.raise_for_status()
        tree = etree.parse(BytesIO(r_pre.content))

        negated_concepts = set()
        for arc in tree.xpath("//link:presentationArc", namespaces={"link": "http://www.xbrl.org/2003/linkbase"}):
            if "negatedLabel" in (arc.get("preferredLabel") or ""):
                to_label = arc.get("{http://www.w3.org/1999/xlink}to")
                loc = tree.xpath(f"//link:loc[@xlink:label='{to_label}']", namespaces={"link": "http://www.xbrl.org/2003/linkbase", "xlink": "http://www.w3.org/1999/xlink"})
                if loc:
                    href = loc[0].get("{http://www.w3.org/1999/xlink}href")
                    if href and "#" in href:
                        concept = href.split("#")[-1].replace("_", ":")
                        negated_concepts.add(concept)
        print(f"✅ Found {len(negated_concepts)} concepts with negated labels.")
        with _CACHE_LOCK:
            _NEGATED_LABELS_CACHE[key] = set(negated_concepts)
        return negated_concepts
    except Exception as e:
        print(f"❌ Error in get_negated_label_concepts: {e}")
        return set()


# In[ ]:


# === CONFIG & SETUP ==========================================
# === Helper: Get concept roles from presentation document for target filing ===

def get_concept_roles_from_presentation(cik, accession_number, headers):
    """
    Returns a dictionary mapping concept names (e.g. 'us-gaap:Assets') to the role(s)
    they appear under in the presentation tree (from .pre.xml).
    """
    
    def normalize_role_uri(uri):
        if not uri or "/role/" not in uri:
            return None
        return uri.split("/role/")[-1]
    
    acc_nodash = accession_number.replace("-", "")
    base_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/"
    index_url = base_url + "index.json"
    key = _cache_key(cik, accession_number)

    with _CACHE_LOCK:
        cached = _CONCEPT_ROLES_CACHE.get(key)
    if cached is not None:
        return _clone_concept_roles(cached)

    try:
        r_index = http_get(index_url, headers=headers, rate_limited=True)
        r_index.raise_for_status()
        index_data = r_index.json()
        items = index_data.get("directory", {}).get("item", [])
        pre_file = next((item["name"] for item in items if "pre" in item["name"].lower() and item["name"].endswith(".xml")), None)
        if not pre_file:
            print(f"⚠️ No .pre.xml found for {accession_number}")
            return {}

        pre_url = base_url + pre_file
        print(f"🔗 Downloading .pre.xml from: {pre_url}")
        r_pre = http_get(pre_url, headers=headers, rate_limited=True)
        r_pre.raise_for_status()
        tree = etree.parse(BytesIO(r_pre.content))

        ns = {
            "link": "http://www.xbrl.org/2003/linkbase",
            "xlink": "http://www.w3.org/1999/xlink"
        }

        concept_roles = {}

        for presentationLink in tree.xpath("//link:presentationLink", namespaces=ns):
            roleURI = presentationLink.get("{http://www.w3.org/1999/xlink}role")
            normalized_role = normalize_role_uri(roleURI)

            # Build label → concept map from <loc> elements
            label_to_concept = {}
            for loc in presentationLink.xpath(".//link:loc", namespaces=ns):
                label = loc.get("{http://www.w3.org/1999/xlink}label")
                href = loc.get("{http://www.w3.org/1999/xlink}href")
                if label and href and "#" in href:
                    concept = href.split("#")[-1].replace("_", ":")
                    label_to_concept[label] = concept

            # Link concepts via <presentationArc> to the role
            for arc in presentationLink.xpath(".//link:presentationArc", namespaces=ns):
                to_label = arc.get("{http://www.w3.org/1999/xlink}to")
                if to_label in label_to_concept:
                    concept = label_to_concept[to_label]
                    concept_roles.setdefault(concept, []).append(normalized_role)

        print(f"✅ Extracted {len(concept_roles)} concept → role mappings from .pre.xml")
        with _CACHE_LOCK:
            _CONCEPT_ROLES_CACHE[key] = _clone_concept_roles(concept_roles)
        return concept_roles

    except Exception as e:
        print(f"❌ Error in get_concept_roles_from_presentation: {e}")
        return {}


# In[ ]:

