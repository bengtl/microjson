// viewer2d/js/LayerPanel.js

const LAYER_LABELS = {
    cells: "Cell Boundaries",
    nuclei: "Nucleus Boundaries",
    transcripts: "Transcripts",
};

export const LayerPanel = {
    init(metadata, rasterLayer, vectorGridLayer, hiddenLayers) {
        const container = document.getElementById("layer-toggles");
        container.innerHTML = "";

        // DAPI raster toggle
        this._addToggle(container, "DAPI Image", "#888888", true, (checked) => {
            if (checked) rasterLayer.setOpacity(1);
            else rasterLayer.setOpacity(0);
        });

        // Vector layer toggles
        metadata.vectors.layers.forEach((layerDef) => {
            this._addToggle(
                container,
                LAYER_LABELS[layerDef.id] || layerDef.id,
                layerDef.color,
                true,
                (checked) => {
                    if (checked) {
                        hiddenLayers.delete(layerDef.id);
                    } else {
                        hiddenLayers.add(layerDef.id);
                    }
                    vectorGridLayer.redraw();
                },
                layerDef.type === "point" ? "circle" : "square"
            );
        });

        // Gene filter — collect gene names from hover events
        this._setupGeneFilter(vectorGridLayer);
    },

    _addToggle(container, label, color, checked, onChange, shape = "square") {
        const wrapper = document.createElement("label");
        wrapper.className = "layer-toggle";
        wrapper.style.setProperty("--layer-color", color);

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = checked;
        checkbox.addEventListener("change", () => onChange(checkbox.checked));

        const swatch = document.createElement("span");
        swatch.className = "layer-swatch";
        swatch.style.background = color;
        if (shape === "circle") swatch.style.borderRadius = "50%";

        const text = document.createTextNode(label);

        wrapper.appendChild(checkbox);
        wrapper.appendChild(swatch);
        wrapper.appendChild(text);
        container.appendChild(wrapper);
    },

    _setupGeneFilter(vectorGridLayer) {
        const select = document.getElementById("gene-filter");
        if (!vectorGridLayer) {
            select.style.display = "none";
            return;
        }

        // Collect gene names progressively from mouseover events
        const genes = new Set();
        vectorGridLayer.on("mouseover", (e) => {
            const name = e.layer.properties && e.layer.properties.gene_name;
            if (name && !genes.has(name)) {
                genes.add(name);
                // Rebuild dropdown
                const current = select.value;
                select.innerHTML = '<option value="">All genes</option>';
                [...genes].sort().forEach((g) => {
                    const opt = document.createElement("option");
                    opt.value = g;
                    opt.textContent = g;
                    select.appendChild(opt);
                });
                select.value = current;
            }
        });

        select.addEventListener("change", () => {
            vectorGridLayer.redraw();
        });
    },
};
