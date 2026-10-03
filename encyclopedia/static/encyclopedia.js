// Shared behaviour: the starburst seal, and the index page's search and quirk filters.
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
  var active = {};

  function apply() {
    var q = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    var wanted = Object.keys(active).filter(function (k) { return active[k]; });
    var shown = 0;
    cards.forEach(function (card) {
      var hay = (card.getAttribute('data-search') || '').toLowerCase();
      var quirks = (card.getAttribute('data-quirks') || '').split(' ');
      var ok = q.every(function (w) { return hay.indexOf(w) !== -1; }) &&
               wanted.every(function (w) { return quirks.indexOf(w) !== -1; });
      card.classList.toggle('hidden', !ok);
      if (ok) shown++;
    });
    if (count) count.textContent = shown + ' of ' + cards.length + ' families';
  }
  search.addEventListener('input', apply);
  document.querySelectorAll('#filters [data-filter]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var key = btn.getAttribute('data-filter');
      active[key] = !active[key];
      btn.classList.toggle('active', !!active[key]);
      apply();
    });
  });
  apply();
})();
