/* Auto-build a desktop-only sticky table-of-contents side panel from the
   post's <h2> sections. Mirrors the RL explainer's panel. Hidden on phones
   via CSS (.toc-side display:none until min-width:1200px). */
(function () {
  var art = document.querySelector('article');
  if (!art) return;
  var heads = [].slice.call(art.querySelectorAll('h2'));
  if (heads.length < 3) return;          // not worth a TOC

  function slug(t) {
    return t.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40);
  }

  var nav = document.createElement('nav');
  nav.className = 'toc-side';
  var lbl = document.createElement('span');
  lbl.className = 'lbl';
  lbl.textContent = 'Contents';
  nav.appendChild(lbl);
  var ol = document.createElement('ol');
  nav.appendChild(ol);

  var map = {};
  heads.forEach(function (h, i) {
    var text = h.textContent.replace(/^\s*\d+\s*/, '').trim();   // drop the leading section number
    if (!h.id) {
      var base = slug(text) || ('sec-' + i), id = base, n = 1;
      while (document.getElementById(id)) id = base + '-' + (n++);
      h.id = id;
    }
    h.style.scrollMarginTop = '78px';                            // clear the sticky header on jump
    var li = document.createElement('li');
    var a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = text;
    li.appendChild(a);
    ol.appendChild(li);
    map[h.id] = a;
  });
  document.body.appendChild(nav);

  // scroll-spy: highlight the section currently in view
  if ('IntersectionObserver' in window) {
    var links = [].slice.call(ol.querySelectorAll('a'));
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          links.forEach(function (l) { l.classList.remove('active'); });
          var a = map[e.target.id];
          if (a) a.classList.add('active');
        }
      });
    }, { rootMargin: '-10% 0px -75% 0px', threshold: 0 });
    heads.forEach(function (h) { obs.observe(h); });
  }
})();
