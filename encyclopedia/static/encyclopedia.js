// Shared behaviour: the starburst seal, and the index page's search and its org, quirk and vision encoder filters.
(function () {
  // A 32-spike starburst as a clip-path, computed once so every seal is the same polygon.
  var pts = [];
  for (var k = 0; k < 64; k++) {
    var a = (k / 64) * Math.PI * 2, r = k % 2 === 0 ? 50 : 42;
    pts.push((50 + r * Math.cos(a)).toFixed(2) + '% ' + (50 + r * Math.sin(a)).toFixed(2) + '%');
  }
  var poly = 'polygon(' + pts.join(', ') + ')';
  document.querySelectorAll('.seal').forEach(function (el) { el.style.clipPath = poly; });

  var search = document.getElementById('search');
  if (!search) return;
  var cards = Array.prototype.slice.call(document.querySelectorAll('#cards .card'));
  var count = document.getElementById('count');
  // the filters on: quirk and vision-encoder slugs a card must all carry, and orgs of which a card must be one
  var active = {}, orgs = {};

  function on(set) { return Object.keys(set).filter(function (k) { return set[k]; }); }
  function apply() {
    var q = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    var wanted = on(active), fromOrgs = on(orgs);
    var shown = 0;
    cards.forEach(function (card) {
      var hay = (card.getAttribute('data-search') || '').toLowerCase();
      var quirks = (card.getAttribute('data-quirks') || '').split(' ');
      var ok = q.every(function (w) { return hay.indexOf(w) !== -1; }) &&
               wanted.every(function (w) { return quirks.indexOf(w) !== -1; }) &&
               (!fromOrgs.length || fromOrgs.indexOf(card.getAttribute('data-org')) !== -1);
      card.classList.toggle('hidden', !ok);
      if (ok) shown++;
    });
    if (count) count.textContent = shown + ' of ' + cards.length + ' families';
  }
  search.addEventListener('input', apply);
  function toggles(selector, attr, set) {
    document.querySelectorAll(selector).forEach(function (btn) {
      btn.addEventListener('click', function () {
        var key = btn.getAttribute(attr);
        set[key] = !set[key];
        btn.classList.toggle('active', !!set[key]);
        btn.setAttribute('aria-pressed', set[key] ? 'true' : 'false');
        apply();
      });
    });
  }
  toggles('#filters [data-filter]', 'data-filter', active);
  toggles('#filters [data-org-filter]', 'data-org-filter', orgs);
  apply();
})();
