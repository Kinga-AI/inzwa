

CORE_REF ?= v0.1.0
CORE_DIR ?= ../kinga-core

sync-core:
\t@if [ ! -d $(CORE_DIR) ]; then echo 'missing core'; exit 1; fi
\tcp $(CORE_DIR)/lang/registry.yaml ./langpacks/REGISTRY.yaml

