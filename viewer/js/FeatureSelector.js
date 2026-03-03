/**
 * Feature selector sidebar — searchable checkbox list of brain regions.
 *
 * Loads features.json and renders a filterable list. Fires a callback
 * whenever the selection changes with the set of selected feature names.
 */

export class FeatureSelector {
    /**
     * @param {HTMLElement} container - DOM element to render into
     * @param {function(Set<string>): void} onSelectionChange - callback with selected names
     */
    constructor(container, onSelectionChange) {
        this.container = container;
        this.onSelectionChange = onSelectionChange;
        this.features = {};          // name → {color, acronym, ccf_id, tiles}
        this.selected = new Set();
        this.checkboxes = new Map();  // name → checkbox element
        this._allItems = [];         // all list item elements
    }

    async init(featuresUrl) {
        const resp = await fetch(featuresUrl);
        const data = await resp.json();
        this.features = data.features;
        this.selected.clear();
        this.checkboxes.clear();
        this._allItems = [];
        this._render();
    }

    _render() {
        this.container.innerHTML = '';

        // Search input
        const search = document.createElement('input');
        search.type = 'text';
        search.placeholder = 'Search regions...';
        search.className = 'feature-search';
        search.addEventListener('input', () => this._filter(search.value));
        this.container.appendChild(search);

        // Toolbar
        const toolbar = document.createElement('div');
        toolbar.className = 'feature-toolbar';

        const count = document.createElement('span');
        count.className = 'feature-count';
        count.id = 'feature-count';
        count.textContent = `0 / ${Object.keys(this.features).length}`;
        toolbar.appendChild(count);

        const selectAllBtn = document.createElement('button');
        selectAllBtn.textContent = 'All';
        selectAllBtn.title = 'Select all visible';
        selectAllBtn.addEventListener('click', () => this._selectAllVisible());
        toolbar.appendChild(selectAllBtn);

        const clearBtn = document.createElement('button');
        clearBtn.textContent = 'Clear';
        clearBtn.title = 'Clear selection';
        clearBtn.addEventListener('click', () => this._clearAll());
        toolbar.appendChild(clearBtn);

        this.container.appendChild(toolbar);

        // List
        const list = document.createElement('div');
        list.className = 'feature-list';

        const names = Object.keys(this.features).sort();
        for (const name of names) {
            const feat = this.features[name];
            const item = document.createElement('label');
            item.className = 'feature-item';
            item.dataset.name = name.toLowerCase();

            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.addEventListener('change', () => this._toggle(name, cb.checked));
            this.checkboxes.set(name, cb);

            const swatch = document.createElement('span');
            swatch.className = 'feature-swatch';
            swatch.style.backgroundColor = feat.color || '#888';

            const label = document.createElement('span');
            label.className = 'feature-name';
            label.textContent = name;
            label.title = `${feat.acronym || ''} (${feat.tiles.length} tiles)`;

            item.appendChild(cb);
            item.appendChild(swatch);
            item.appendChild(label);
            list.appendChild(item);
            this._allItems.push(item);
        }

        this.container.appendChild(list);
    }

    _toggle(name, checked) {
        if (checked) {
            this.selected.add(name);
        } else {
            this.selected.delete(name);
        }
        this._updateCount();
        this.onSelectionChange(this.selected);
    }

    _filter(query) {
        const q = query.toLowerCase().trim();
        for (const item of this._allItems) {
            item.style.display = (!q || item.dataset.name.includes(q)) ? '' : 'none';
        }
    }

    _selectAllVisible() {
        for (const item of this._allItems) {
            if (item.style.display !== 'none') {
                const name = item.querySelector('.feature-name').textContent;
                const cb = this.checkboxes.get(name);
                if (cb && !cb.checked) {
                    cb.checked = true;
                    this.selected.add(name);
                }
            }
        }
        this._updateCount();
        this.onSelectionChange(this.selected);
    }

    _clearAll() {
        for (const [name, cb] of this.checkboxes) {
            cb.checked = false;
            this.selected.delete(name);
        }
        this._updateCount();
        this.onSelectionChange(this.selected);
    }

    _updateCount() {
        const el = document.getElementById('feature-count');
        if (el) el.textContent = `${this.selected.size} / ${Object.keys(this.features).length}`;
    }
}
