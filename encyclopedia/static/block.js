// Draws one decoder block from the selected checkpoint's schema and wires every part, in the SVG and in the
// model strip above it, to the side card: hover shows a part's nnterp name, click pins it. Every part
// sits in a group carrying its role class (the stylesheet colours it from there), and the card takes
// the role of the part it shows. A family whose blocks come in several shapes (a hybrid's mixers, a
// dense MLP before the mixtures) has one shape drawn at a time: the slider's block picks it.
//
// The page holds every checkpoint's data (#checkpoints-json) and every checkpoint's panes (.pane, one
// copy per distinct rendering, data-ckpts naming the checkpoints it is for). The selector in the hero
// picks one: the panes swap, the block and its slider redraw, and on a vision-language checkpoint the
// vision encoder's block is drawn with the same code, its node ids prefixed "v:". The choice is the URL hash,
// #ckpt=<repo id>, read on load.
(function () {
  var store = JSON.parse(document.getElementById('checkpoints-json').textContent);
  var mainSvg = document.getElementById('block-svg');
  var mainCard = document.getElementById('node-card');
  var NS = 'http://www.w3.org/2000/svg';
  var TOWER = 'v:';

  // The selected checkpoint: its data, the text block's schema, and every node text (the tower's prefixed).
  var selected = null, data = null, schema = null, nodes = {};

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
  // A sublayer with more than six chips grows a chip row at a time; a mixture's box holds a panel
  // each for the router, the routed experts and the shared expert, with their values as chips.
  var W = 1040, SX = 150, TOP = 80, ROW = 220, RET = 990, BASE = 104, PANEL = 48, PGAP = 8;
  var PRE = { x: 220, w: 190, h: 48 }, SUB = { x: 440, w: 300 }, POST = { x: 770, w: 190, h: 48 };
  // a norm on the stream after an add (a post-LN block) sits on the line below the add, SN taller
  var SNORM = { w: 170, h: 44, gap: 22 }, SN = 60;
  var PARTS = [['router', 'router'], ['experts', 'routed experts'], ['shared', 'shared expert']];
  function keyOf(sub) { return sub.key || sub.host; }
  function partsOf(sub) {
    return PARTS.filter(function (p) { return sub.interior.some(function (v) { return v.part === p[0]; }); });
  }
  function heightOf(sub) {
    var chips = sub.interior ? sub.interior.length : 0;
    if (sub.kind === 'moe' && chips) return 60 + partsOf(sub).length * (PANEL + PGAP);
    var rows = Math.ceil(chips / 3);
    return rows > 2 ? 58 + rows * 24 : BASE;
  }
  function shapesOf(s) {
    return s.shapes || [{ subs: s.sublayers.map(function (_, k) { return k; }), plus: 'plus' }];
  }

  // What draw() is drawing into: the SVG, the schema, and the prefix its node ids take.
  var ctx = { svg: mainSvg, schema: null, prefix: '' };
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]);
    (parent || ctx.svg).appendChild(e);
    return e;
  }
  function text(parent, x, y, str, cls, anchor) {
    var t = el('text', { x: x, y: y, 'class': cls, 'text-anchor': anchor || 'start' }, parent);
    t.textContent = str;
    return t;
  }
  function group(id, role, cls) { return el('g', { 'data-node': ctx.prefix + id, 'class': 'role-' + role + (cls ? ' ' + cls : '') }); }
  // the role a node's colours come from: a sublayer's parts take the sublayer's, norms are norms,
  // the stream, the add, the root's strip nodes and the image's path are the stream
  function roleOf(id) {
    var tower = id.indexOf(TOWER) === 0, s = tower ? (data.tower && data.tower.schema) : schema;
    var parts = (tower ? id.slice(TOWER.length) : id).split('.');
    if (['sub', 'contrib', 'interior', 'moe'].indexOf(parts[0]) !== -1) return (s && s.roles[parts[1]]) || 'mlp';
    if (parts[0] === 'norm' || id === 'strip.norm' || id === TOWER + 'path.norm') return 'norm';
    return 'stream';
  }
  function marker(id, color) {
    var defs = ctx.svg.querySelector('defs') || el('defs', {});
    var m = el('marker', { id: id, markerWidth: 10, markerHeight: 10, refX: 8, refY: 5, orient: 'auto', markerUnits: 'userSpaceOnUse' }, defs);
    el('path', { d: 'M0,0 L9,5 L0,10 Z', fill: color }, m);
  }
  function chip(id, role, x, y, w, short) {
    var g = group(id, role, 'chipnode');
    el('rect', { x: x, y: y, width: w, height: 18, 'class': 'chip-r' }, g);
    text(g, x + w / 2, y + 13, short, 'chip-t', 'middle');
  }

  // One shape of block from schema `s` into `target`: its sublayers in order, between the stream's two ends.
  function draw(target, s, shape, prefix) {
    ctx = { svg: target, schema: s, prefix: prefix || '' };
    var svg = target, all = s.sublayers, parallel = s.topology === 'parallel', P = ctx.prefix;
    var arr = 'arr' + (P ? '-tower' : ''), arrow = 'url(#' + arr + ')';
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    marker(arr, getComputedStyle(document.documentElement).getPropertyValue('--ink').trim() || '#3A2516');
    var subs = shape.subs.map(function (k) { return all[k]; }), n = subs.length;
    // a sublayer marked parallel_with_next and the next one branch from one stream point and join at one add
    var first = subs.map(function (sub, k) { return !parallel && !!sub.parallel_with_next && k + 1 < n; });
    var second = subs.map(function (_, k) { return k > 0 && first[k - 1]; });
    // a box taller than the base moves the rows after it down by its extra height
    var ext = subs.map(function (sub) { return (heightOf(sub) - BASE) / 2; });
    var rowY = [], joinY = [], y = TOP + ext[0];
    var sn = subs.map(function (sub) { return sub.stream_norm ? SN : 0; });
    subs.forEach(function (_, k) {
      rowY.push(y);
      y += (parallel || first[k] ? 130 : ROW) + ext[k] + (k + 1 < n ? ext[k + 1] : 0) + sn[k];
    });
    subs.forEach(function (_, k) {
      joinY.push(parallel ? rowY[n - 1] + 110 + ext[n - 1] : first[k] ? rowY[k + 1] + 110 + ext[k + 1] : rowY[k] + 140 + ext[k]);
    });
    var H = joinY[n - 1] + 60 + sn[n - 1];
    // where the stream resumes below an add: under its stream norm, where it has one
    function below(k) { return sn[k] ? joinY[k] + SNORM.gap + SNORM.h : joinY[k]; }
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    var mids = shape.mids || subs.slice(1).map(function (_, k) { return 'stream.mid.' + k; });

    // -- the stream -----------------------------------------------------------------
    var topY = 24, botY = H - 26;
    var segments = [[topY, rowY[0], 'stream.input']];
    subs.forEach(function (sub, k) { if (sn[k]) segments.push([joinY[k], joinY[k] + SNORM.gap, sub.stream_norm_in]); });
    if (!parallel) for (var k = 0; k < n - 1; k++) if (!first[k]) segments.push([below(k), rowY[k + 1], mids[k]]);
    segments.push([below(n - 1), botY, 'stream.output']);
    // the stream between a branch and its join carries the input of that sublayer's add: same name as the segment above
    var full = el('g', { 'class': 'role-stream streamline' });
    el('line', { x1: SX, y1: topY, x2: SX, y2: botY, 'class': 'stream', 'marker-end': arrow }, full);
    segments.forEach(function (seg) {
      var g = group(seg[2], 'stream');
      el('line', { x1: SX, y1: seg[0], x2: SX, y2: seg[1], 'class': 'stream' }, g);
      el('rect', { x: SX - 18, y: seg[0], width: 36, height: Math.max(seg[1] - seg[0], 1), 'class': 'hit' }, g);
    });
    text(full, SX + 16, topY + 6, (P ? 'vision.' : '') + 'layers[i].input', 'label-role');
    text(full, SX + 16, botY - 2, 'layer_output', 'label-role');

    // -- sublayers ------------------------------------------------------------------
    subs.forEach(function (sub, k) {
      var y = rowY[k], exitY = parallel ? rowY[0] : second[k] ? rowY[k - 1] : y, jY = joinY[k], key = keyOf(sub), role = s.roles[key] || 'mlp';
      var h = heightOf(sub);
      // branch out of the stream
      el('path', { 'class': 'edge', d: parallel || second[k]
        ? 'M' + SX + ',' + exitY + ' H' + 190 + ' V' + y + ' H' + (sub.pre_norm ? PRE.x : SUB.x)
        : 'M' + SX + ',' + exitY + ' H' + (sub.pre_norm ? PRE.x : SUB.x), 'marker-end': arrow });
      el('circle', { cx: SX, cy: exitY, r: 5, fill: 'var(--stream-deep)' });
      if (sub.pre_norm) {
        var gp = group(sub.pre_norm_node || 'norm.' + sub.pre_norm, 'norm');
        el('rect', { x: PRE.x, y: y - PRE.h / 2, width: PRE.w, height: PRE.h, 'class': 'box box-norm' }, gp);
        text(gp, PRE.x + PRE.w / 2, y - 4, 'norm', 'label-dim', 'middle');
        text(gp, PRE.x + PRE.w / 2, y + 12, sub.pre_norm, 'label-sm', 'middle');
        el('path', { 'class': 'edge', d: 'M' + (PRE.x + PRE.w) + ',' + y + ' H' + SUB.x, 'marker-end': arrow });
      }
      var gs = group('sub.' + key, role);
      el('rect', { x: SUB.x, y: y - h / 2, width: SUB.w, height: h, 'class': 'box box-sub' }, gs);
      var hasChips = sub.interior && sub.interior.length;
      var top = y - h / 2;
      text(gs, SUB.x + 14, top + 30, sub.label, 'label');
      text(gs, SUB.x + 14, top + 46, sub.host, 'label-sm');
      var detail = hasChips
        ? text(gs, SUB.x + SUB.w - 12, top + 46, '', 'label-dim', 'end')
        : text(gs, SUB.x + 14, top + 78, '', 'label-dim');
      detail.setAttribute('data-variant-for', key);
      // The room the line has: beside the host's name when chips take the rows below, else the box's width.
      detail.setAttribute('data-room', hasChips ? SUB.w - 40 - sub.host.length * 6.8 : SUB.w - 28);
      if (hasChips && sub.kind === 'moe') {
        // a panel per part, each a hover node of its own, drawn over the box so it takes its own hover
        partsOf(sub).forEach(function (part, j) {
          var py = top + 56 + j * (PANEL + PGAP), px = SUB.x + 12, pw = SUB.w - 24;
          var gpart = group('moe.' + key + '.' + part[0], role);
          el('rect', { x: px, y: py, width: pw, height: PANEL, 'class': 'box box-part' }, gpart);
          text(gpart, px + 8, py + 14, part[1], 'label-dim');
          var size = part[0] === 'router' ? sub.moe.scoring
            : part[0] === 'experts' ? sub.moe.top_k + ' of ' + sub.moe.num_experts + ' per token' : 'every token';
          text(gpart, px + pw - 8, py + 14, size, 'label-dim', 'end');
          sub.interior.filter(function (v) { return v.part === part[0]; }).forEach(function (v, c) {
            chip('interior.' + key + '.' + v.name, role, px + 8 + c * 88, py + 22, 80, v.short);
          });
        });
      } else if (hasChips) {
        sub.interior.forEach(function (v, j) {
          chip('interior.' + key + '.' + v.name, role, SUB.x + 14 + (j % 3) * 94, top + 56 + Math.floor(j / 3) * 24, 86, v.short);
        });
      }
      var outX = SUB.x + SUB.w;
      if (sub.post_norm) {
        el('path', { 'class': 'edge', d: 'M' + outX + ',' + y + ' H' + POST.x, 'marker-end': arrow });
        var gq = group(sub.post_norm_node || 'norm.' + sub.post_norm, 'norm');
        el('rect', { x: POST.x, y: y - POST.h / 2, width: POST.w, height: POST.h, 'class': 'box box-norm' }, gq);
        text(gq, POST.x + POST.w / 2, y - 4, 'norm', 'label-dim', 'middle');
        text(gq, POST.x + POST.w / 2, y + 12, sub.post_norm, 'label-sm', 'middle');
        outX = POST.x + POST.w;
      }
      // the contribution: back into the stream
      // In a parallel block every contribution meets the stream at the one add, each on its own
      // return path, the first sublayer's outermost, so no two edges or labels share a line.
      var lane = parallel ? (n - 1 - k) : first[k] ? 1 : 0, retX = RET - lane * 24, inY = jY - lane * 26;
      var gc2 = group('contrib.' + key, role);
      var d = 'M' + outX + ',' + y + ' H' + retX + ' V' + inY + ' H' + (SX + 16);
      el('path', { 'class': 'edge-contrib', d: d, 'marker-end': arrow }, gc2);
      el('path', { 'class': 'hit', d: d, 'stroke-width': 18, fill: 'none', stroke: 'transparent' }, gc2);
      text(gc2, retX - 8, inY - 10, sub.contribution, 'label-role', 'end');
      // the add
      if (parallel ? k === n - 1 : !first[k]) {
        var gplus = group(shape.plus, 'stream');
        el('circle', { cx: SX, cy: jY, r: 14, 'class': 'plus' }, gplus);
        text(gplus, SX, jY + 8, '+', 'plus-sign', 'middle');
      }
      // a norm on the stream after the add, over the line
      if (sub.stream_norm) {
        var gn = group('norm.' + sub.stream_norm, 'norm'), ny = jY + SNORM.gap;
        el('rect', { x: SX - SNORM.w / 2, y: ny, width: SNORM.w, height: SNORM.h, 'class': 'box box-norm' }, gn);
        text(gn, SX, ny + 18, 'norm', 'label-dim', 'middle');
        text(gn, SX, ny + 34, sub.stream_norm, 'label-sm', 'middle');
      }
    });
    // SVG has no z-index: the last child is on top. The contribution edges' wide hit paths were
    // drawn after the boxes and covered the chips along a box's lower edge, so the boxes and
    // their chips go last.
    Array.prototype.slice.call(svg.querySelectorAll(['sub.', 'interior.', 'moe.'].map(function (p) {
      return 'g[data-node^="' + P + p + '"]';
    }).join(', '))).forEach(function (g) { svg.appendChild(g); });
  }

  // -- the side cards ---------------------------------------------------------------
  // The block's card shows the text model's parts; the tower's sub-section has a card of its own.
  var pinned = null;
  function towerPane() { return document.querySelector('.pane[data-pane="tower"]:not([hidden])'); }
  function cardFor(id) {
    var pane = id.indexOf(TOWER) === 0 ? towerPane() : null;
    return (pane && pane.querySelector('.node-card')) || mainCard;
  }
  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }
  function inline(s) { return esc(s).replace(/`([^`]+)`/g, '<code>$1</code>'); }
  function layer() { return document.getElementById('layer') ? document.getElementById('layer').value : 'i'; }
  function show(id) {
    var node = nodes[id];
    if (!node) return;
    var card = cardFor(id);
    // the slider picks a text block; the tower's block is any of them
    var expr = id.indexOf(TOWER) === 0 ? node.expr : node.expr.replace(/\[i\]/g, '[' + layer() + ']');
    var html = '<p class="micro eyebrow">' + esc(node.eyebrow) + '</p>' +
      '<p class="expr">' + esc(expr) + '</p>';
    if (node.layout) html += '<p class="body-md"><span class="chip chip-layout">' + esc(node.layout) + '</span> <span class="mono dim">[' + esc(node.dims) + ']</span></p>';
    html += '<p class="body-md">' + inline(node.desc) + '</p>';
    if (node.where) html += '<p class="body-md dim">Read at ' + inline(node.where) + '.</p>';
    if (node.extra) html += '<p class="body-md">' + inline(node.extra) + '</p>';
    if (node.condition) html += '<div class="cond"><b>' + (node.condition.kind === 'eager' ? 'needs eager' : 'conditional') + '</b>' + esc(node.condition.reason) + '</div>';
    html += '<p class="mono dim">' + (pinned === id ? 'pinned · click again to release' : 'click to pin') + '</p>';
    card.querySelector('.card-body').innerHTML = html;
    card.style.setProperty('--role', 'var(--' + roleOf(id) + ')');
    card.style.setProperty('--role-deep', 'var(--' + roleOf(id) + '-deep)');
  }
  function hot(id, on) {
    document.querySelectorAll('[data-node="' + id + '"]').forEach(function (e) { e.classList.toggle('hot', on); e.classList.toggle('active', on); });
  }
  function unpin() {
    if (!pinned) return;
    hot(pinned, false);
    cardFor(pinned).classList.remove('pinned');
    pinned = null;
  }
  var current = null;
  document.addEventListener('mouseover', function (ev) {
    var t = ev.target.closest && ev.target.closest('[data-node]');
    if (!t || !t.getAttribute('data-node')) return;
    var id = t.getAttribute('data-node');
    if (current && current !== id) hot(current, false);
    current = id; hot(id, true);
    if (!pinned || cardFor(pinned) !== cardFor(id)) show(id);
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
    if (pinned === id) { unpin(); show(id); return; }
    unpin();
    pinned = id; cardFor(id).classList.add('pinned'); hot(id, true); show(id);
  });

  // -- the layer slider -------------------------------------------------------------
  // A tick takes its block's shape's colour on a family with several shapes, else its layer type's.
  var slider = document.getElementById('layer'), label = document.getElementById('layer-label'), variant = document.getElementById('layer-variant');
  var ticks = document.getElementById('ticks'), identity = document.querySelector('#identity code');
  var shapes, shapeOf, types, drawnShape = null;
  function buildTicks() {
    var kinds = [];
    if (types) types.forEach(function (t) { if (kinds.indexOf(t) === -1) kinds.push(t); });
    while (ticks.firstChild) ticks.removeChild(ticks.firstChild);
    for (var i = 0; i < schema.num_layers; i++) {
      var tick = document.createElement('i');
      if (shapeOf) tick.className = 't' + Math.min(shapeOf[i], 4);
      else if (types) tick.className = 't' + kinds.indexOf(types[i]);
      tick.title = 'block ' + i + (types ? ' · ' + types[i] : '') + (shapeOf ? ' · ' + shapes[shapeOf[i]].label : '');
      tick.setAttribute('role', 'button');
      tick.setAttribute('tabindex', '0');
      tick.setAttribute('aria-label', tick.title);
      tick.setAttribute('data-block', i);
      ticks.appendChild(tick);
    }
  }
  // A tick selects its block, as moving the slider to it does.
  function pickBlock(tick) {
    if (!tick || !tick.hasAttribute('data-block')) return;
    slider.value = tick.getAttribute('data-block');
    update();
  }
  ticks.addEventListener('click', function (e) { pickBlock(e.target.closest('[data-block]')); });
  ticks.addEventListener('keydown', function (e) {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    pickBlock(e.target.closest('[data-block]'));
  });
  function update() {
    var i = parseInt(slider.value, 10);
    var s = shapeOf ? shapeOf[i] : 0;
    if (s !== drawnShape) {
      draw(mainSvg, schema, shapes[s]);
      drawnShape = s;
      identity.innerHTML = shapeOf ? shapes[s].identity_html : data.identity_html;
      if (pinned && pinned.indexOf(TOWER) !== 0) hot(pinned, true);
    }
    label.textContent = 'i = ' + i;
    var t = types ? types[i] : null;
    variant.textContent = (t ? t.replace(/_/g, ' ') : '') + (shapeOf ? (t ? ' · ' : '') + shapes[s].label : '');
    Array.prototype.forEach.call(ticks.children, function (c, j) { c.classList.toggle('cur', j === i); });
    shapes[s].subs.forEach(function (k) {
      var sub = schema.sublayers[k], d = mainSvg.querySelector('[data-variant-for="' + keyOf(sub) + '"]');
      if (!d) return;
      fit(d, (t && sub.variants[t]) ? sub.variants[t] : sub.detail);
    });
    if (pinned && pinned.indexOf(TOWER) !== 0) show(pinned);
  }
  slider.addEventListener('input', update);

  // -- the checkpoint -----------------------------------------------------------------
  var button = document.getElementById('ckpt-button'), list = document.getElementById('ckpt-list');
  var options = Array.prototype.slice.call(list.querySelectorAll('[role="option"]'));
  function usable(o) { return o.getAttribute('aria-disabled') !== 'true'; }
  function optionOf(id) { return options.filter(function (o) { return o.getAttribute('data-ckpt') === id; })[0]; }

  function select(id) {
    if (!store.checkpoints[id]) return;
    unpin();
    selected = id; data = store.checkpoints[id]; schema = data.schema;
    nodes = data.nodes;
    // the panes: each shows when it is the selected checkpoint's
    document.querySelectorAll('.pane').forEach(function (p) {
      p.hidden = p.getAttribute('data-ckpts').split(' ').indexOf(id) === -1;
    });
    // the block and its slider, from this checkpoint's blocks
    shapes = shapesOf(schema); shapeOf = schema.shape_of; types = schema.layer_types; drawnShape = null;
    slider.max = schema.num_layers - 1;
    if (parseInt(slider.value, 10) > schema.num_layers - 1) slider.value = 0;
    buildTicks();
    update();
    drawTower();
    // the selector, its Hub link and the colophon say which
    var option = optionOf(id);
    options.forEach(function (o) { o.setAttribute('aria-selected', o === option ? 'true' : 'false'); });
    document.getElementById('ckpt-name').textContent = option ? option.getAttribute('data-name') : id;
    document.getElementById('ckpt-eye').hidden = !(option && option.hasAttribute('data-vision'));
    var hub = document.getElementById('ckpt-hub');
    hub.href = data.url;
    hub.title = id + ' on the Hugging Face Hub';
    hub.setAttribute('aria-label', hub.title);
    document.getElementById('colophon-reference').textContent = id;
  }

  // The vision encoder's block, in the pane now shown, once its fold is open (a closed fold has no layout to fit
  // the detail lines in).
  function drawTower() {
    var pane = towerPane(), fold = pane && pane.querySelector('details');
    if (!pane || !data.tower || !data.tower.schema || (fold && !fold.open)) return;  // a blockless encoder draws no block
    var ts = data.tower.schema, tsvg = pane.querySelector('.tower-svg');
    draw(tsvg, ts, shapesOf(ts)[0], TOWER);
    ts.sublayers.forEach(function (sub) {
      var d = tsvg.querySelector('[data-variant-for="' + keyOf(sub) + '"]');
      if (d) fit(d, sub.detail);
    });
  }
  document.addEventListener('toggle', function (ev) {
    if (ev.target.open && ev.target.closest && ev.target.closest('.pane[data-pane="tower"]')) drawTower();
  }, true);

  function fromHash() {
    var m = /^#ckpt=(.+)$/.exec(location.hash);
    if (!m) return null;
    var id = decodeURIComponent(m[1]);
    return store.checkpoints[id] ? id : null;
  }

  // The list: a button opens it; arrows move, Enter or Space picks, Escape closes, a click picks.
  var active = null;
  function setActive(o) {
    if (active) active.classList.remove('is-active');
    active = o;
    if (!o) { list.removeAttribute('aria-activedescendant'); return; }
    o.classList.add('is-active');
    list.setAttribute('aria-activedescendant', o.id);
    if (o.scrollIntoView) o.scrollIntoView({ block: 'nearest' });
  }
  function open() {
    list.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    setActive(optionOf(selected) || options.filter(usable)[0]);
    list.focus();
  }
  function close(refocus) {
    list.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    setActive(null);
    if (refocus) button.focus();
  }
  function pick(o) {
    if (!o || !usable(o)) return;
    var id = o.getAttribute('data-ckpt');
    close(true);
    if (id === selected) return;
    select(id);
    if (history.replaceState) history.replaceState(null, '', '#ckpt=' + id);
    else location.hash = 'ckpt=' + id;
  }
  button.addEventListener('click', function () { if (list.hidden) open(); else close(true); });
  button.addEventListener('keydown', function (ev) {
    if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') { ev.preventDefault(); open(); }
  });
  list.addEventListener('keydown', function (ev) {
    var choices = options.filter(usable), at = choices.indexOf(active);
    if (ev.key === 'ArrowDown') { ev.preventDefault(); setActive(choices[Math.min(at + 1, choices.length - 1)]); }
    else if (ev.key === 'ArrowUp') { ev.preventDefault(); setActive(choices[Math.max(at - 1, 0)]); }
    else if (ev.key === 'Home') { ev.preventDefault(); setActive(choices[0]); }
    else if (ev.key === 'End') { ev.preventDefault(); setActive(choices[choices.length - 1]); }
    else if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); pick(active); }
    else if (ev.key === 'Escape') { ev.preventDefault(); close(true); }
    else if (ev.key === 'Tab') close(false);
  });
  list.addEventListener('click', function (ev) {
    var o = ev.target.closest && ev.target.closest('[role="option"]');
    if (o) pick(o);
  });
  list.addEventListener('mousemove', function (ev) {
    var o = ev.target.closest && ev.target.closest('[role="option"]');
    if (o && usable(o) && o !== active) setActive(o);
  });
  document.addEventListener('click', function (ev) {
    if (!list.hidden && !ev.target.closest('#ckpt')) close(false);
  });
  window.addEventListener('hashchange', function () {
    var id = fromHash();
    if (id && id !== selected) select(id);
  });

  select(fromHash() || store.default);
})();
