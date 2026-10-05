// Draws one decoder block from the schema the page embeds and wires every part, in the SVG and in the
// model strip above it, to the side card: hover shows a part's nnterp name, click pins it. Every part
// sits in a group carrying its role class (the stylesheet colours it from there), and the card takes
// the role of the part it shows.
(function () {
  var schema = JSON.parse(document.getElementById('block-schema').textContent);
  var nodes = JSON.parse(document.getElementById('nodes-json').textContent);
  var svg = document.getElementById('block-svg');
  var card = document.getElementById('node-card');
  var NS = 'http://www.w3.org/2000/svg';

  // A detail line longer than its room ends in an ellipsis; the node card carries it whole.
  function fit(node, str) {
    var room = parseFloat(node.getAttribute('data-room'));
    node.textContent = str;
    while (str.length > 1 && node.getComputedTextLength() > room) {
      str = str.slice(0, -1);
      node.textContent = str.replace(/[\s,;:]+$/, '') + '…';
    }
  }

  // -- geometry ---------------------------------------------------------------------
  var W = 1040, SX = 150, TOP = 80, ROW = 220, RET = 990;
  var PRE = { x: 220, w: 190, h: 48 }, SUB = { x: 440, w: 300, h: 104 }, POST = { x: 770, w: 190, h: 48 };
  var subs = schema.sublayers, n = subs.length, parallel = schema.topology === 'parallel';
  var rowY = subs.map(function (_, k) { return TOP + k * (parallel ? 130 : ROW); });
  var joinY = subs.map(function (_, k) { return parallel ? TOP + (n - 1) * 130 + 110 : rowY[k] + 140; });
  var H = joinY[n - 1] + 60;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);

  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]);
    (parent || svg).appendChild(e);
    return e;
  }
  function text(parent, x, y, str, cls, anchor) {
    var t = el('text', { x: x, y: y, 'class': cls, 'text-anchor': anchor || 'start' }, parent);
    t.textContent = str;
    return t;
  }
  function group(id, role, cls) { return el('g', { 'data-node': id, 'class': 'role-' + role + (cls ? ' ' + cls : '') }); }
  // the role a node's colours come from: a sublayer's parts take the sublayer's, norms are norms,
  // the stream, the add and the root's strip nodes are the stream
  function roleOf(id) {
    var parts = id.split('.');
    if (parts[0] === 'sub' || parts[0] === 'contrib' || parts[0] === 'interior') return schema.roles[parts[1]] || 'mlp';
    if (parts[0] === 'norm' || id === 'strip.norm') return 'norm';
    return 'stream';
  }
  function marker(id, color) {
    var defs = svg.querySelector('defs') || el('defs', {});
    var m = el('marker', { id: id, markerWidth: 10, markerHeight: 10, refX: 8, refY: 5, orient: 'auto', markerUnits: 'userSpaceOnUse' }, defs);
    el('path', { d: 'M0,0 L9,5 L0,10 Z', fill: color }, m);
  }
  marker('arr', getComputedStyle(document.documentElement).getPropertyValue('--ink').trim() || '#3A2516');

  // -- the stream -------------------------------------------------------------------
  var topY = 24, botY = H - 26;
  var segments = [[topY, parallel ? rowY[0] : rowY[0], 'stream.input']];
  if (!parallel) for (var k = 0; k < n - 1; k++) segments.push([joinY[k], rowY[k + 1], 'stream.mid.' + k]);
  segments.push([joinY[n - 1], botY, 'stream.output']);
  // the stream between a branch and its join carries the input of that sublayer's add: same name as the segment above
  var full = el('g', { 'class': 'role-stream streamline' });
  el('line', { x1: SX, y1: topY, x2: SX, y2: botY, 'class': 'stream', 'marker-end': 'url(#arr)' }, full);
  segments.forEach(function (s) {
    var g = group(s[2], 'stream');
    el('line', { x1: SX, y1: s[0], x2: SX, y2: s[1], 'class': 'stream' }, g);
    el('rect', { x: SX - 18, y: s[0], width: 36, height: Math.max(s[1] - s[0], 1), 'class': 'hit' }, g);
  });
  text(full, SX + 16, topY + 6, 'layers[i].input', 'label-role');
  text(full, SX + 16, botY - 2, 'layer_output', 'label-role');

  // -- sublayers --------------------------------------------------------------------
  subs.forEach(function (sub, k) {
    var y = rowY[k], exitY = parallel ? TOP : y, jY = joinY[k], role = schema.roles[sub.host] || 'mlp';
    // branch out of the stream
    var branch = el('path', { 'class': 'edge', d: parallel
      ? 'M' + SX + ',' + exitY + ' H' + 190 + ' V' + y + ' H' + (sub.pre_norm ? PRE.x : SUB.x)
      : 'M' + SX + ',' + exitY + ' H' + (sub.pre_norm ? PRE.x : SUB.x), 'marker-end': 'url(#arr)' });
    el('circle', { cx: SX, cy: exitY, r: 5, fill: 'var(--stream-deep)' });
    if (sub.pre_norm) {
      var gp = group('norm.' + sub.pre_norm, 'norm');
      el('rect', { x: PRE.x, y: y - PRE.h / 2, width: PRE.w, height: PRE.h, 'class': 'box box-norm' }, gp);
      text(gp, PRE.x + PRE.w / 2, y - 4, 'norm', 'label-dim', 'middle');
      text(gp, PRE.x + PRE.w / 2, y + 12, sub.pre_norm, 'label-sm', 'middle');
      el('path', { 'class': 'edge', d: 'M' + (PRE.x + PRE.w) + ',' + y + ' H' + SUB.x, 'marker-end': 'url(#arr)' });
    }
    var gs = group('sub.' + sub.host, role);
    el('rect', { x: SUB.x, y: y - SUB.h / 2, width: SUB.w, height: SUB.h, 'class': 'box box-sub' }, gs);
    var hasChips = sub.interior && sub.interior.length;
    var top = y - SUB.h / 2;
    text(gs, SUB.x + 14, top + 30, sub.label, 'label');
    text(gs, SUB.x + 14, top + 46, sub.host, 'label-sm');
    var detail = hasChips
      ? text(gs, SUB.x + SUB.w - 12, top + 46, '', 'label-dim', 'end')
      : text(gs, SUB.x + 14, top + 78, '', 'label-dim');
    detail.setAttribute('data-variant-for', sub.host);
    // The room the line has: beside the host's name when chips take the rows below, else the box's width.
    detail.setAttribute('data-room', hasChips ? SUB.w - 40 - sub.host.length * 6.8 : SUB.w - 28);
    if (hasChips) {
      var cw = 86, ch = 18, gap = 8, x0 = SUB.x + 14;
      sub.interior.forEach(function (v, j) {
        var cx = x0 + (j % 3) * (cw + gap), cy = top + 56 + Math.floor(j / 3) * (ch + 6);
        var gc = group('interior.' + sub.host + '.' + v.name, role, 'chipnode');
        el('rect', { x: cx, y: cy, width: cw, height: ch, 'class': 'chip-r' }, gc);
        text(gc, cx + cw / 2, cy + 13, v.short, 'chip-t', 'middle');
      });
    }
    var outX = SUB.x + SUB.w;
    if (sub.post_norm) {
      el('path', { 'class': 'edge', d: 'M' + outX + ',' + y + ' H' + POST.x, 'marker-end': 'url(#arr)' });
      var gq = group('norm.' + sub.post_norm, 'norm');
      el('rect', { x: POST.x, y: y - POST.h / 2, width: POST.w, height: POST.h, 'class': 'box box-norm' }, gq);
      text(gq, POST.x + POST.w / 2, y - 4, 'norm', 'label-dim', 'middle');
      text(gq, POST.x + POST.w / 2, y + 12, sub.post_norm, 'label-sm', 'middle');
      outX = POST.x + POST.w;
    }
    // the contribution: back into the stream
    // In a parallel block every contribution meets the stream at the one add, each on its own
    // return path, the first sublayer's outermost, so no two edges or labels share a line.
    var lane = parallel ? (n - 1 - k) : 0, retX = RET - lane * 24, inY = jY - lane * 26;
    var gc2 = group('contrib.' + sub.host, role);
    var d = 'M' + outX + ',' + y + ' H' + retX + ' V' + inY + ' H' + (SX + 16);
    el('path', { 'class': 'edge-contrib', d: d, 'marker-end': 'url(#arr)' }, gc2);
    el('path', { 'class': 'hit', d: d, 'stroke-width': 18, fill: 'none', stroke: 'transparent' }, gc2);
    text(gc2, retX - 8, inY - 10, sub.contribution, 'label-role', 'end');
    // the add
    if (!parallel || k === n - 1) {
      var gplus = group('plus', 'stream');
      el('circle', { cx: SX, cy: jY, r: 14, 'class': 'plus' }, gplus);
      text(gplus, SX, jY + 8, '+', 'plus-sign', 'middle');
    }
  });

  // -- the side card ----------------------------------------------------------------
  var pinned = null;
  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }
  function inline(s) { return esc(s).replace(/`([^`]+)`/g, '<code>$1</code>'); }
  function layer() { return document.getElementById('layer') ? document.getElementById('layer').value : 'i'; }
  function show(id) {
    var node = nodes[id];
    if (!node) return;
    var expr = node.expr.replace(/\[i\]/g, '[' + layer() + ']');
    var html = '<p class="micro eyebrow">' + esc(node.eyebrow) + '</p>' +
      '<p class="expr">' + esc(expr) + '</p>';
    if (node.layout) html += '<p class="body-md"><span class="chip chip-layout">' + esc(node.layout) + '</span> <span class="mono dim">[' + esc(node.dims) + ']</span></p>';
    html += '<p class="body-md">' + inline(node.desc) + '</p>';
    if (node.where) html += '<p class="body-md dim">Read at ' + inline(node.where) + '.</p>';
    if (node.extra) html += '<p class="body-md">' + inline(node.extra) + '</p>';
    if (node.condition) html += '<div class="cond"><b>' + (node.condition.kind === 'eager' ? 'needs eager' : 'conditional') + '</b>' + esc(node.condition.reason) + '</div>';
    html += '<p class="mono dim">' + (pinned ? 'pinned · click again to release' : 'click to pin') + '</p>';
    card.querySelector('.card-body').innerHTML = html;
    card.style.setProperty('--role', 'var(--' + roleOf(id) + ')');
    card.style.setProperty('--role-deep', 'var(--' + roleOf(id) + '-deep)');
  }
  function hot(id, on) {
    document.querySelectorAll('[data-node="' + id + '"]').forEach(function (e) { e.classList.toggle('hot', on); e.classList.toggle('active', on); });
  }
  var current = null;
  document.addEventListener('mouseover', function (ev) {
    var t = ev.target.closest && ev.target.closest('[data-node]');
    if (!t || !t.getAttribute('data-node')) return;
    var id = t.getAttribute('data-node');
    if (current && current !== id) hot(current, false);
    current = id; hot(id, true);
    if (!pinned) show(id);
  });
  document.addEventListener('mouseout', function (ev) {
    var t = ev.target.closest && ev.target.closest('[data-node]');
    if (!t) return;
    var id = t.getAttribute('data-node');
    if (id !== pinned) hot(id, false);
    if (current === id) current = null;
  });
  document.addEventListener('click', function (ev) {
    var t = ev.target.closest && ev.target.closest('[data-node]');
    if (!t || t.tagName === 'A') return;
    var id = t.getAttribute('data-node');
    if (!id || !nodes[id]) return;
    if (pinned === id) { pinned = null; card.classList.remove('pinned'); show(id); return; }
    if (pinned) hot(pinned, false);
    pinned = id; card.classList.add('pinned'); hot(id, true); show(id);
  });

  // -- the layer slider -------------------------------------------------------------
  var slider = document.getElementById('layer'), label = document.getElementById('layer-label'), variant = document.getElementById('layer-variant');
  var ticks = document.getElementById('ticks'), types = schema.layer_types, kinds = [];
  if (types) types.forEach(function (t) { if (kinds.indexOf(t) === -1) kinds.push(t); });
  if (ticks) {
    for (var i = 0; i < schema.num_layers; i++) {
      var tick = document.createElement('i');
      if (types) tick.className = 't' + kinds.indexOf(types[i]);
      tick.title = 'block ' + i + (types ? ' · ' + types[i] : '');
      ticks.appendChild(tick);
    }
  }
  function update() {
    var i = parseInt(slider.value, 10);
    label.textContent = 'i = ' + i;
    var t = types ? types[i] : null;
    variant.textContent = t ? t.replace(/_/g, ' ') : '';
    if (ticks) Array.prototype.forEach.call(ticks.children, function (c, j) { c.classList.toggle('cur', j === i); });
    subs.forEach(function (sub) {
      var d = svg.querySelector('[data-variant-for="' + sub.host + '"]');
      if (!d) return;
      fit(d, (t && sub.variants[t]) ? sub.variants[t] : sub.detail);
    });
    if (pinned) show(pinned);
  }
  if (slider) { slider.addEventListener('input', update); update(); }
})();
