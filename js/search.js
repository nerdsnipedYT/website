/* Client-side blog search + tag filtering for the index page.
   - Full-text search across every post (loads search-index.json; falls back
     to card title/description/tags if it can't).
   - Clickable tags (on cards, in the tag bar, or via ?tag= in the URL) filter
     the list to matching posts. */
(function () {
  var wrap = document.getElementById('cards');
  if (!wrap) return;                                   // index page only

  var cards = [].slice.call(wrap.querySelectorAll('.card'));
  var searchEl = document.getElementById('search');
  var tagbar = document.getElementById('tagbar');
  var noresults = document.getElementById('noresults');
  var activefilter = document.getElementById('activefilter');

  function slug(s) {
    return s.toLowerCase().trim().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  }

  // sort newest-first by data-date (stable for equal dates -> keeps DOM order)
  cards.map(function (c, i) { return { c: c, d: c.getAttribute('data-date') || '', i: i }; })
    .sort(function (a, b) { return a.d < b.d ? 1 : a.d > b.d ? -1 : a.i - b.i; })
    .forEach(function (o) { wrap.appendChild(o.c); });
  cards = [].slice.call(wrap.querySelectorAll('.card'));

  var items = cards.map(function (c) {
    var tags = (c.getAttribute('data-tags') || '').split('|')
      .map(function (x) { return x.trim(); }).filter(Boolean);
    return {
      el: c, url: c.getAttribute('href'), tags: tags,
      tagslugs: tags.map(slug), text: (c.textContent || '').toLowerCase(), body: ''
    };
  });

  var tagLabel = {};
  items.forEach(function (it) { it.tags.forEach(function (t) { tagLabel[slug(t)] = t; }); });

  var state = { q: '', tag: '' };
  try { state.tag = new URLSearchParams(location.search).get('tag') || ''; } catch (e) {}

  // build the tag bar from the union of all post tags
  if (tagbar) {
    var seen = {}, all = [];
    items.forEach(function (it) {
      it.tags.forEach(function (t) { var s = slug(t); if (!seen[s]) { seen[s] = 1; all.push(t); } });
    });
    var allBtn = document.createElement('button');
    allBtn.className = 'tag-chip'; allBtn.textContent = 'All'; allBtn.dataset.tag = '';
    tagbar.appendChild(allBtn);
    all.forEach(function (t) {
      var b = document.createElement('button');
      b.className = 'tag-chip'; b.textContent = t; b.dataset.tag = slug(t);
      tagbar.appendChild(b);
    });
    tagbar.addEventListener('click', function (e) {
      var b = e.target.closest('.tag-chip'); if (!b) return;
      setTag(b.dataset.tag);
    });
  }

  // tags shown inside cards filter instead of opening the post
  wrap.addEventListener('click', function (e) {
    var chip = e.target.closest('.tag-chip'); if (!chip) return;
    e.preventDefault(); e.stopPropagation();
    setTag(chip.dataset.tag || slug(chip.textContent));
  });

  function setTag(t) { state.tag = (state.tag === t) ? '' : t; apply(); }

  function apply() {
    var terms = state.q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    var visible = 0;
    items.forEach(function (it) {
      var okTag = state.tag === '' || it.tagslugs.indexOf(state.tag) >= 0;
      var hay = it.text + ' ' + it.body;
      var okQ = terms.every(function (tm) { return hay.indexOf(tm) >= 0; });
      var show = okTag && okQ;
      it.el.hidden = !show; if (show) visible++;
    });
    if (noresults) noresults.hidden = visible > 0;
    if (tagbar) [].slice.call(tagbar.children).forEach(function (b) {
      b.classList.toggle('active', (b.dataset.tag || '') === state.tag);
    });
    if (activefilter) {
      if (state.tag) {
        var label = (tagLabel[state.tag] || state.tag).replace(/</g, '&lt;');
        activefilter.innerHTML = '';
        var lbl = document.createElement('span'); lbl.textContent = 'Filtered by tag:';
        var btn = document.createElement('button'); btn.type = 'button'; btn.title = 'Clear filter';
        btn.innerHTML = label + ' <span aria-hidden="true">✕</span>';
        btn.addEventListener('click', function () { setTag(''); });
        activefilter.appendChild(lbl); activefilter.appendChild(btn);
        activefilter.hidden = false;
      } else {
        activefilter.hidden = true; activefilter.innerHTML = '';
      }
    }
    try {
      var u = new URL(location.href);
      if (state.tag) u.searchParams.set('tag', state.tag); else u.searchParams.delete('tag');
      history.replaceState(null, '', u);
    } catch (e) {}
  }

  if (searchEl) searchEl.addEventListener('input', function () { state.q = searchEl.value; apply(); });

  apply();   // show something immediately (title/desc/tag search)

  // upgrade to full-text search once the index loads
  fetch('search-index.json').then(function (r) { return r.json(); }).then(function (idx) {
    var map = {}; idx.forEach(function (e) { map[e.url] = e; });
    items.forEach(function (it) {
      var e = map[it.url];
      if (e) it.body = ((e.text || '') + ' ' + (e.title || '') + ' ' +
        (e.desc || '') + ' ' + (e.tags || []).join(' ')).toLowerCase();
    });
    apply();
  }).catch(function () {});
})();
